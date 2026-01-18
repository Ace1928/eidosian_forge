import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_sample_from_amplitudes():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1), cirq.X(q1))
    sim = cirq.Simulator(seed=1)
    result = sim.sample_from_amplitudes(circuit, {}, sim._prng, repetitions=100)
    assert 40 < result[1] < 60
    assert 40 < result[2] < 60
    assert 0 not in result
    assert 3 not in result