import random
import numpy as np
import pytest
import cirq
def _random_single_MS_effect():
    t = random.random()
    s = np.sin(t)
    c = np.cos(t)
    return cirq.dot(cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)), np.array([[c, 0, 0, -1j * s], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [-1j * s, 0, 0, c]]), cirq.kron(cirq.testing.random_unitary(2), cirq.testing.random_unitary(2)))