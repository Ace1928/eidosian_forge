import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_simulate_noise_with_terminal_measurements():
    q = cirq.LineQubit(0)
    circuit1 = cirq.Circuit(cirq.measure(q))
    circuit2 = circuit1 + cirq.I(q)
    simulator = cirq.DensityMatrixSimulator(noise=cirq.X)
    result1 = simulator.run(circuit1, repetitions=10)
    result2 = simulator.run(circuit2, repetitions=10)
    assert result1 == result2