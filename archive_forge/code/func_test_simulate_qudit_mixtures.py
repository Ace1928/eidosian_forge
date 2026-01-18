import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('dtype, split', itertools.product([np.complex64, np.complex128], [True, False]))
def test_simulate_qudit_mixtures(dtype: Type[np.complexfloating], split: bool):
    q0 = cirq.LineQid(0, 3)
    simulator = cirq.Simulator(dtype=dtype, split_untangled_states=split)
    mixture = _TestMixture([cirq.XPowGate(dimension=3) ** 0, cirq.XPowGate(dimension=3), cirq.XPowGate(dimension=3) ** 2])
    circuit = cirq.Circuit(mixture(q0), cirq.measure(q0))
    counts = {0: 0, 1: 0, 2: 0}
    for _ in range(300):
        result = simulator.simulate(circuit, qubit_order=[q0])
        meas = result.measurements['q(0) (d=3)'][0]
        counts[meas] += 1
        np.testing.assert_almost_equal(result.final_state_vector, np.array([meas == 0, meas == 1, meas == 2]))
    assert counts[0] < 160 and counts[0] > 40
    assert counts[1] < 160 and counts[1] > 40
    assert counts[2] < 160 and counts[2] > 40