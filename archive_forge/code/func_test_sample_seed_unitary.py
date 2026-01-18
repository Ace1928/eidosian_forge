import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_sample_seed_unitary():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.X(q) ** 0.2, cirq.measure(q))
    result = cirq.sample(circuit, repetitions=10, seed=1234)
    measurements = result.measurements['q']
    assert np.all(measurements == [[False], [False], [False], [False], [False], [False], [False], [False], [True], [False]])