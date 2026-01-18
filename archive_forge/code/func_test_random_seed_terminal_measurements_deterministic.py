import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_random_seed_terminal_measurements_deterministic():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.X(a) ** 0.5, cirq.measure(a, key='a'))
    sim = cirq.DensityMatrixSimulator(seed=1234)
    result1 = sim.run(circuit, repetitions=30)
    result2 = sim.run(circuit, repetitions=30)
    assert np.all(result1.measurements['a'] == [[0], [1], [0], [1], [1], [0], [0], [1], [1], [1], [0], [1], [1], [1], [0], [1], [1], [0], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [0], [1]])
    assert np.all(result2.measurements['a'] == [[1], [0], [1], [0], [1], [1], [0], [1], [0], [1], [0], [0], [0], [1], [1], [1], [0], [1], [0], [1], [0], [1], [1], [0], [1], [1], [1], [1], [1], [1]])