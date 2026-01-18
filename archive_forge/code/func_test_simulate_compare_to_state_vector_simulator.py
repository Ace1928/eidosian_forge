import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype,circuit', itertools.product([np.complex64, np.complex128], [cirq.testing.random_circuit(cirq.LineQubit.range(4), 5, 0.9) for _ in range(20)]))
def test_simulate_compare_to_state_vector_simulator(dtype: Type[np.complexfloating], circuit):
    qubits = cirq.LineQubit.range(4)
    pure_result = cirq.Simulator(dtype=dtype).simulate(circuit, qubit_order=qubits).density_matrix_of()
    mixed_result = cirq.DensityMatrixSimulator(dtype=dtype).simulate(circuit, qubit_order=qubits).final_density_matrix
    assert mixed_result.shape == (16, 16)
    np.testing.assert_almost_equal(mixed_result, pure_result, decimal=6)