from __future__ import annotations
import logging
import numpy as np
from qiskit.circuit import Gate, ParameterExpression, Qubit
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.circuit.reset import Reset
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.target import Target
from .base_padding import BasePadding
def _pre_runhook(self, dag: DAGCircuit):
    super()._pre_runhook(dag)
    num_pulses = len(self._dd_sequence)
    if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
        raise TranspilerError('DD runs on physical circuits only.')
    if self._spacing is None:
        mid = 1 / num_pulses
        end = mid / 2
        self._spacing = [end] + [mid] * (num_pulses - 1) + [end]
    elif sum(self._spacing) != 1 or any((a < 0 for a in self._spacing)):
        raise TranspilerError('The spacings must be given in terms of fractions of the slack period and sum to 1.')
    if num_pulses != 1:
        if num_pulses % 2 != 0:
            raise TranspilerError('DD sequence must contain an even number of gates (or 1).')
        noop = np.eye(2)
        for gate in self._dd_sequence:
            noop = noop.dot(gate.to_matrix())
        if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
            raise TranspilerError('The DD sequence does not make an identity operation.')
        self._sequence_phase = np.angle(noop[0][0])
    for qarg, _ in enumerate(dag.qubits):
        for gate in self._dd_sequence:
            if not self.__gate_supported(gate, qarg):
                self._no_dd_qubits.add(qarg)
                logger.debug('No DD on qubit %d as gate %s is not supported on it', qarg, gate.name)
                break
    for physical_index, qubit in enumerate(dag.qubits):
        if not self.__is_dd_qubit(physical_index):
            continue
        sequence_lengths = []
        for index, gate in enumerate(self._dd_sequence):
            try:
                params = self._resolve_params(gate)
                gate_length = dag.calibrations[gate.name][(physical_index,), params].duration
                if gate_length % self._alignment != 0:
                    raise TranspilerError(f'Pulse gate {gate.name} with length non-multiple of {self._alignment} is not acceptable in {self.__class__.__name__} pass.')
            except KeyError:
                gate_length = self._durations.get(gate, physical_index)
            sequence_lengths.append(gate_length)
            gate = gate.to_mutable()
            self._dd_sequence[index] = gate
            gate.duration = gate_length
        self._dd_sequence_lengths[qubit] = sequence_lengths