import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_unitary():
    rs = np.random.RandomState(52)
    for _ in range(10):
        qubits = cirq.LineQubit.range(5)
        circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8, random_state=rs)
        operator = _random_pauli_string(qubits, rs)
        circuit_sand = ccq.circuit_for_expectation_value(circuit, operator)
        u_tn = ccq.tensor_unitary(circuit_sand, qubits)
        u_cirq = cirq.unitary(circuit_sand)
        np.testing.assert_allclose(u_tn, u_cirq, atol=1e-06)