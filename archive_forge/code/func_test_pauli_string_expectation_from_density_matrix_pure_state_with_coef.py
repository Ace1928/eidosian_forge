import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_pauli_string_expectation_from_density_matrix_pure_state_with_coef():
    qs = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qs)}
    circuit = cirq.Circuit(cirq.X(qs[1]), cirq.H(qs[2]), cirq.X(qs[3]), cirq.H(qs[3]))
    state_vector = circuit.final_state_vector(qubit_order=qs, ignore_terminal_measurements=False, dtype=np.complex128)
    rho = np.outer(state_vector, np.conj(state_vector))
    z0z1 = cirq.Z(qs[0]) * cirq.Z(qs[1]) * 0.123
    z0z2 = cirq.Z(qs[0]) * cirq.Z(qs[2]) * -1
    z1x2 = -cirq.Z(qs[1]) * cirq.X(qs[2])
    for state in [rho, rho.reshape((2, 2, 2, 2, 2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), -0.123)
        np.testing.assert_allclose(z0z2.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(z1x2.expectation_from_density_matrix(state, q_map), 1)