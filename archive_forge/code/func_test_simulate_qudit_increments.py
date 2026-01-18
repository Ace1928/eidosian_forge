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
def test_simulate_qudit_increments(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    for b0 in [0, 1]:
        for b1 in [0, 1, 2]:
            circuit = cirq.Circuit((cirq.X ** b0)(q0), (cirq.XPowGate(dimension=3)(q1),) * b1, cirq.measure(q0), cirq.measure(q1))
            result = simulator.simulate(circuit)
            np.testing.assert_equal(result.measurements, {'q(0) (d=2)': [b0], 'q(1) (d=3)': [b1]})
            expected_density_matrix = np.zeros(shape=(6, 6))
            expected_density_matrix[b0 * 3 + b1, b0 * 3 + b1] = 1.0
            np.testing.assert_allclose(result.final_density_matrix, expected_density_matrix)