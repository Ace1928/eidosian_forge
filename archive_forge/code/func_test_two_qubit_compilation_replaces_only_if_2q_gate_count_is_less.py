from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_two_qubit_compilation_replaces_only_if_2q_gate_count_is_less():

    class ExampleTargetGateset(cirq.TwoQubitCompilationTargetGateset):

        def __init__(self):
            super().__init__(cirq.X, cirq.CNOT)

        def _decompose_two_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
            q0, q1 = op.qubits
            return [cirq.X.on_each(q0, q1), cirq.CNOT(q0, q1)] * 10

        def _decompose_single_qubit_operation(self, op: 'cirq.Operation', _) -> DecomposeResult:
            return cirq.X(*op.qubits) if op.gate == cirq.Y else NotImplemented
    q = cirq.LineQubit.range(2)
    ops = [cirq.Y.on_each(*q), cirq.CNOT(*q), cirq.Z.on_each(*q)]
    c_orig = cirq.Circuit(ops)
    c_expected = cirq.Circuit(cirq.X.on_each(*q), ops[-2:])
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=ExampleTargetGateset())
    cirq.testing.assert_same_circuits(c_new, c_expected)