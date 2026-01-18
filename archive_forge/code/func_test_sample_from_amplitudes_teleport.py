import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_sample_from_amplitudes_teleport():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.H(q1), cirq.CNOT(q1, q2), cirq.X(q0) ** sympy.Symbol('t'), cirq.CNOT(q0, q1), cirq.H(q0), cirq.CNOT(q1, q2), cirq.CZ(q0, q2), cirq.H(q0), cirq.H(q1))
    sim = cirq.Simulator(seed=1)
    result_a = sim.sample_from_amplitudes(circuit, {'t': 1}, sim._prng, repetitions=100)
    assert result_a == {1: 100}
    result_b = sim.sample_from_amplitudes(circuit, {'t': 0.5}, sim._prng, repetitions=100)
    assert 40 < result_b[0] < 60
    assert 40 < result_b[1] < 60
    result_c = sim.sample_from_amplitudes(circuit, {'t': 0.25}, sim._prng, repetitions=100)
    assert 80 < result_c[0]
    assert result_c[1] < 20