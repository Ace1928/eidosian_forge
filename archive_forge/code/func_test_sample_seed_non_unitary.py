import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_sample_seed_non_unitary():
    q = cirq.NamedQubit('q')
    circuit = cirq.Circuit(cirq.depolarize(0.5).on(q), cirq.measure(q))
    result = cirq.sample(circuit, repetitions=10, seed=1234)
    assert np.all(result.measurements['q'] == [[False], [False], [False], [True], [True], [False], [False], [True], [True], [True]])