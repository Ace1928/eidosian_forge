import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_random_seed_mixture_deterministic():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.depolarize(0.9).on(a), cirq.depolarize(0.9).on(a), cirq.depolarize(0.9).on(a), cirq.depolarize(0.9).on(a), cirq.depolarize(0.9).on(a), cirq.measure(a, key='a'))
    sim = cirq.Simulator(seed=1234)
    result = sim.run(circuit, repetitions=30)
    assert np.all(result.measurements['a'] == [[1], [0], [0], [0], [1], [0], [0], [1], [1], [1], [1], [1], [0], [1], [0], [0], [0], [0], [0], [1], [0], [1], [1], [0], [1], [1], [1], [1], [1], [0]])