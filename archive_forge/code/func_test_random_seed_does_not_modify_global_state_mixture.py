import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_random_seed_does_not_modify_global_state_mixture():
    a = cirq.NamedQubit('a')
    circuit = cirq.Circuit(cirq.depolarize(0.5).on(a), cirq.measure(a))
    sim = cirq.Simulator(seed=1234)
    result1 = sim.run(circuit, repetitions=50)
    sim = cirq.Simulator(seed=1234)
    _ = np.random.random()
    _ = random.random()
    result2 = sim.run(circuit, repetitions=50)
    assert result1 == result2