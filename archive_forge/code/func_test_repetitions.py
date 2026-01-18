import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('repetitions', (0, 1, 100))
def test_repetitions(repetitions):
    a = cirq.LineQubit(0)
    c = cirq.Circuit(cirq.H(a), cirq.measure(a, key='m'))
    r = cirq.sample(c, repetitions=repetitions)
    samples = r.data['m'].to_numpy()
    assert samples.shape == (repetitions,)
    assert np.issubdtype(samples.dtype, np.integer)