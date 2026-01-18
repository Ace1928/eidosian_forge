import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_to_z_basis_ops():
    x0 = np.array([1, 1]) / np.sqrt(2)
    x1 = np.array([1, -1]) / np.sqrt(2)
    y0 = np.array([1, 1j]) / np.sqrt(2)
    y1 = np.array([1, -1j]) / np.sqrt(2)
    z0 = np.array([1, 0])
    z1 = np.array([0, 1])
    q0, q1, q2, q3, q4, q5 = _make_qubits(6)
    pauli_string = cirq.PauliString({q0: cirq.X, q1: cirq.X, q2: cirq.Y, q3: cirq.Y, q4: cirq.Z, q5: cirq.Z})
    circuit = cirq.Circuit(pauli_string.to_z_basis_ops())
    initial_state = cirq.kron(x0, x1, y0, y1, z0, z1, shape_len=1)
    z_basis_state = circuit.final_state_vector(initial_state=initial_state, ignore_terminal_measurements=False, dtype=np.complex64)
    expected_state = np.zeros(2 ** 6)
    expected_state[21] = 1
    cirq.testing.assert_allclose_up_to_global_phase(z_basis_state, expected_state, rtol=1e-07, atol=1e-07)