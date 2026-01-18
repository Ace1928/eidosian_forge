import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_state_vector_4():
    qubits = cirq.LineQubit.range(4)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=100, op_density=0.8)
    psi1 = cirq.final_state_vector(circuit, dtype=np.complex128)
    psi2 = ccq.tensor_state_vector(circuit, qubits)
    np.testing.assert_allclose(psi1, psi2, atol=1e-08)