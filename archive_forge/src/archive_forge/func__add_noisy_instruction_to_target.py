from __future__ import annotations
import warnings
from collections.abc import Iterable
import numpy as np
from qiskit import pulse
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.controlflow import (
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap, Target, InstructionProperties, QubitProperties
from qiskit.providers import Options
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.backend import BackendV2
from qiskit.providers.models import (
from qiskit.qobj import PulseQobjInstruction, PulseLibraryItem
from qiskit.utils import optionals as _optionals
def _add_noisy_instruction_to_target(self, instruction: Instruction, noise_params: tuple[float, ...] | None, calibration_inst_map: InstructionScheduleMap | None) -> None:
    """Add instruction properties to target for specified instruction.

        Args:
            instruction: Instance of instruction to be added to the target
            noise_params: Error and duration noise values/ranges to
                include in instruction properties.
            calibration_inst_map: Instruction schedule map with calibration defaults
        """
    qarg_set = self._coupling_map if instruction.num_qubits > 1 else range(self.num_qubits)
    props = {}
    for qarg in qarg_set:
        try:
            qargs = tuple(qarg)
        except TypeError:
            qargs = (qarg,)
        duration, error = noise_params if len(noise_params) == 2 else (self._rng.uniform(*noise_params[:2]), self._rng.uniform(*noise_params[2:]))
        if calibration_inst_map is not None and instruction.name not in ['reset', 'delay'] and (qarg in calibration_inst_map.qubits_with_instruction(instruction.name)):
            calibration_entry = calibration_inst_map._get_calibration_entry(instruction.name, qargs)
        else:
            calibration_entry = None
        props.update({qargs: InstructionProperties(duration, error, calibration_entry)})
    self._target.add_instruction(instruction, props)
    if calibration_inst_map is not None and instruction.name == 'measure':
        for qarg in calibration_inst_map.qubits_with_instruction(instruction.name):
            try:
                qargs = tuple(qarg)
            except TypeError:
                qargs = (qarg,)
            calibration_entry = calibration_inst_map._get_calibration_entry(instruction.name, qargs)
            for qubit in qargs:
                if qubit < self.num_qubits:
                    self._target[instruction.name][qubit,].calibration = calibration_entry