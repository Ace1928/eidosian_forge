import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_simulate_bit_flips(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    for b0 in [0, 1]:
        for b1 in [0, 1]:
            circuit = cirq.Circuit((cirq.X ** b0)(q0), (cirq.X ** b1)(q1), cirq.measure(q0), cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {'q(0)': [b0], 'q(1)': [b1]})
            expected_density_matrix = np.zeros(shape=(4, 4))
            expected_density_matrix[b0 * 2 + b1, b0 * 2 + b1] = 1.0
            np.testing.assert_equal(result.final_density_matrix, expected_density_matrix)