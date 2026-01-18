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
def test_simulate_moment_steps_qudits(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    circuit = cirq.Circuit(cirq.XPowGate(dimension=2)(q0), cirq.XPowGate(dimension=3)(q1), cirq.reset(q1), cirq.XPowGate(dimension=3)(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    for i, step in enumerate(simulator.simulate_moment_steps(circuit)):
        assert cirq.qid_shape(step) == (2, 3)
        if i == 0:
            np.testing.assert_almost_equal(step.density_matrix(), np.diag([0, 0, 0, 0, 1, 0]))
        elif i == 1:
            np.testing.assert_almost_equal(step.density_matrix(), np.diag([0, 0, 0, 1, 0, 0]))
        else:
            np.testing.assert_almost_equal(step.density_matrix(), np.diag([0, 0, 0, 0, 1, 0]))