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
def test_simulate_noisy_expectation_values(dtype):
    q0 = cirq.LineQubit(0)
    psums = [cirq.Z(q0), cirq.X(q0)]
    c1 = cirq.Circuit(cirq.X(q0), cirq.amplitude_damp(gamma=0.1).on(q0))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.simulate_expectation_values(c1, psums)
    assert cirq.approx_eq(result[0], -0.8, atol=1e-06)
    assert cirq.approx_eq(result[1], 0, atol=1e-06)
    c2 = cirq.Circuit(cirq.H(q0), cirq.depolarize(p=0.3).on(q0))
    result = simulator.simulate_expectation_values(c2, psums)
    assert cirq.approx_eq(result[0], 0, atol=1e-06)
    assert cirq.approx_eq(result[1], 0.6, atol=1e-06)