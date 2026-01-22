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
def _build_generic_target(self):
    """This method generates a :class:`~.Target` instance with
        default qubit, instruction and calibration properties.
        """
    properties = _QUBIT_PROPERTIES
    self._target = Target(description=f'Generic Target with {self._num_qubits} qubits', num_qubits=self._num_qubits, dt=properties['dt'], qubit_properties=[QubitProperties(t1=self._rng.uniform(properties['t1'][0], properties['t1'][1]), t2=self._rng.uniform(properties['t2'][0], properties['t2'][1]), frequency=self._rng.uniform(properties['frequency'][0], properties['frequency'][1])) for _ in range(self._num_qubits)], concurrent_measurements=[list(range(self._num_qubits))])
    calibration_inst_map = None
    if self._calibrate_instructions is not None:
        if isinstance(self._calibrate_instructions, InstructionScheduleMap):
            calibration_inst_map = self._calibrate_instructions
        else:
            defaults = self._generate_calibration_defaults()
            calibration_inst_map = defaults.instruction_schedule_map
    for name in self._basis_gates:
        if name not in self._supported_gates:
            raise QiskitError(f'Provided basis gate {name} is not an instruction in the standard qiskit circuit library.')
        gate = self._supported_gates[name]
        noise_params = self._get_noise_defaults(name, gate.num_qubits)
        self._add_noisy_instruction_to_target(gate, noise_params, calibration_inst_map)
    if self._control_flow:
        self._target.add_instruction(IfElseOp, name='if_else')
        self._target.add_instruction(WhileLoopOp, name='while_loop')
        self._target.add_instruction(ForLoopOp, name='for_loop')
        self._target.add_instruction(SwitchCaseOp, name='switch_case')
        self._target.add_instruction(BreakLoopOp, name='break')
        self._target.add_instruction(ContinueLoopOp, name='continue')