import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
@pytest.mark.parametrize('circuit', [cirq.testing.random_circuit(cirq.LineQubit.range(2), 4, 0.5) for _ in range(5)])
def test_random_same_matrix(circuit):
    a, b = cirq.LineQubit.range(2)
    same = cirq.Circuit(cirq.MatrixGate(circuit.unitary(qubits_that_should_be_present=[a, b])).on(a, b))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, same)
    mutable_circuit = circuit.copy()
    mutable_circuit.append(cirq.measure(a))
    same.append(cirq.measure(a))
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(mutable_circuit, same)