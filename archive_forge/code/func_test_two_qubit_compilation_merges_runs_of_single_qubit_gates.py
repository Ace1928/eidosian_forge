from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_two_qubit_compilation_merges_runs_of_single_qubit_gates():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CNOT(*q), cirq.X(q[0]), cirq.Y(q[0]), cirq.CNOT(*q))
    cirq.testing.assert_same_circuits(cirq.optimize_for_target_gateset(c, gateset=ExampleCXTargetGateset()), cirq.Circuit(cirq.CNOT(*q), cirq.PhasedXZGate(axis_phase_exponent=-0.5, x_exponent=0, z_exponent=-1).on(q[0]), cirq.CNOT(*q)))