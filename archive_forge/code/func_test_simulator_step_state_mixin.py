import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_simulator_step_state_mixin():
    qubits = cirq.LineQubit.range(2)
    args = cirq.StateVectorSimulationState(available_buffer=np.array([0, 1, 0, 0]).reshape((2, 2)), prng=cirq.value.parse_random_state(0), qubits=qubits, initial_state=np.array([0, 1, 0, 0], dtype=np.complex64).reshape((2, 2)), dtype=np.complex64)
    result = cirq.SparseSimulatorStep(sim_state=args, dtype=np.complex64)
    rho = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    np.testing.assert_array_almost_equal(rho, result.density_matrix_of(qubits))
    bloch = np.array([0, 0, -1])
    np.testing.assert_array_almost_equal(bloch, result.bloch_vector_of(qubits[1]))
    assert result.dirac_notation() == '|01⟩'