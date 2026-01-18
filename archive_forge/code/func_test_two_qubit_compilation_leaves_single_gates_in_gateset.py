from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_two_qubit_compilation_leaves_single_gates_in_gateset():
    q = cirq.LineQubit.range(2)
    gateset = ExampleCXTargetGateset()
    c = cirq.Circuit(cirq.X(q[0]) ** 0.5)
    cirq.testing.assert_same_circuits(cirq.optimize_for_target_gateset(c, gateset=gateset), c)
    c = cirq.Circuit(cirq.CNOT(*q[:2]))
    cirq.testing.assert_same_circuits(cirq.optimize_for_target_gateset(c, gateset=gateset), c)