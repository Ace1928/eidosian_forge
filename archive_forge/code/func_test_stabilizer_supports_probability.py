import numpy as np
import pytest
import sympy
import cirq
def test_stabilizer_supports_probability():
    q = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.X(q).with_probability(0.5), cirq.measure(q, key='m'))
    m = np.sum(cirq.StabilizerSampler().sample(c, repetitions=100)['m'])
    assert 5 < m < 95