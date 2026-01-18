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
def test_simulate_expectation_values_qubit_order(dtype):
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q0), cirq.H(q1), cirq.X(q2))
    obs = cirq.X(q0) + cirq.X(q1) - cirq.Z(q2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype)
    result = simulator.simulate_expectation_values(circuit, obs)
    assert cirq.approx_eq(result[0], 3, atol=1e-06)
    result_flipped = simulator.simulate_expectation_values(circuit, obs, qubit_order=[q1, q2, q0])
    assert cirq.approx_eq(result_flipped[0], 3, atol=1e-06)