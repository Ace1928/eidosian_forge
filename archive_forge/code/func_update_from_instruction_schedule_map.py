from a backend
from __future__ import annotations
import itertools
from typing import Optional, List, Any
from collections.abc import Mapping
from collections import defaultdict
import datetime
import io
import logging
import inspect
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.calibration_entries import CalibrationEntry, ScheduleDef
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import QubitProperties  # pylint: disable=unused-import
from qiskit.providers.models.backendproperties import BackendProperties
def update_from_instruction_schedule_map(self, inst_map, inst_name_map=None, error_dict=None):
    """Update the target from an instruction schedule map.

        If the input instruction schedule map contains new instructions not in
        the target they will be added. However, if it contains additional qargs
        for an existing instruction in the target it will error.

        Args:
            inst_map (InstructionScheduleMap): The instruction
            inst_name_map (dict): An optional dictionary that maps any
                instruction name in ``inst_map`` to an instruction object.
                If not provided, instruction is pulled from the standard Qiskit gates,
                and finally custom gate instance is created with schedule name.
            error_dict (dict): A dictionary of errors of the form::

                {gate_name: {qarg: error}}

            for example::

                {'rx': {(0, ): 1.4e-4, (1, ): 1.2e-4}}

            For each entry in the ``inst_map`` if ``error_dict`` is defined
            a when updating the ``Target`` the error value will be pulled from
            this dictionary. If one is not found in ``error_dict`` then
            ``None`` will be used.
        """
    get_calibration = getattr(inst_map, '_get_calibration_entry')
    qiskit_inst_name_map = get_standard_gate_name_mapping()
    if inst_name_map is not None:
        qiskit_inst_name_map.update(inst_name_map)
    for inst_name in inst_map.instructions:
        out_props = {}
        for qargs in inst_map.qubits_with_instruction(inst_name):
            try:
                qargs = tuple(qargs)
            except TypeError:
                qargs = (qargs,)
            try:
                props = self._gate_map[inst_name][qargs]
            except (KeyError, TypeError):
                props = None
            entry = get_calibration(inst_name, qargs)
            if entry.user_provided and getattr(props, '_calibration', None) != entry:
                if self.dt is not None:
                    try:
                        duration = entry.get_schedule().duration * self.dt
                    except UnassignedDurationError:
                        duration = None
                else:
                    duration = None
                props = InstructionProperties(duration=duration, calibration=entry)
            elif props is None:
                continue
            try:
                props.error = error_dict[inst_name][qargs]
            except (KeyError, TypeError):
                pass
            out_props[qargs] = props
        if not out_props:
            continue
        if inst_name not in self._gate_map:
            if inst_name in qiskit_inst_name_map:
                inst_obj = qiskit_inst_name_map[inst_name]
                normalized_props = {}
                for qargs, prop in out_props.items():
                    if len(qargs) != inst_obj.num_qubits:
                        continue
                    normalized_props[qargs] = prop
                self.add_instruction(inst_obj, normalized_props, name=inst_name)
            else:
                qlen = set()
                param_names = set()
                for qargs in inst_map.qubits_with_instruction(inst_name):
                    if isinstance(qargs, int):
                        qargs = (qargs,)
                    qlen.add(len(qargs))
                    cal = getattr(out_props[tuple(qargs)], '_calibration')
                    param_names.add(tuple(cal.get_signature().parameters.keys()))
                if len(qlen) > 1 or len(param_names) > 1:
                    raise QiskitError(f'Schedules for {inst_name} are defined non-uniformly for multiple qubit lengths {qlen}, or different parameter names {param_names}. Provide these schedules with inst_name_map or define them with different names for different gate parameters.')
                inst_obj = Gate(name=inst_name, num_qubits=next(iter(qlen)), params=list(map(Parameter, next(iter(param_names)))))
                self.add_instruction(inst_obj, out_props, name=inst_name)
        else:
            for qargs, prop in out_props.items():
                if qargs not in self._gate_map[inst_name]:
                    continue
                self.update_instruction_properties(inst_name, qargs, prop)