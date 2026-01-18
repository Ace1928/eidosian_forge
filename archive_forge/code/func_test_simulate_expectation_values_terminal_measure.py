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
def test_simulate_expectation_values_terminal_measure(dtype):
    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0))
    obs = cirq.Z(q0)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    with pytest.raises(ValueError):
        _ = simulator.simulate_expectation_values(circuit, obs)
    results = {-1: 0, 1: 0}
    for _ in range(100):
        result = simulator.simulate_expectation_values(circuit, obs, permit_terminal_measurements=True)
        if cirq.approx_eq(result[0], -1, atol=1e-06):
            results[-1] += 1
        if cirq.approx_eq(result[0], 1, atol=1e-06):
            results[1] += 1
    assert results[-1] > 0
    assert results[1] > 0
    assert results[-1] + results[1] == 100
    circuit = cirq.Circuit(cirq.H(q0))
    results = {0: 0}
    for _ in range(100):
        result = simulator.simulate_expectation_values(circuit, obs, permit_terminal_measurements=True)
        if cirq.approx_eq(result[0], 0, atol=1e-06):
            results[0] += 1
    assert results[0] == 100