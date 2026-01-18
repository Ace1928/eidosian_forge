import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_pack_results():
    measurements = [('a', np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])), ('b', np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0]]))]
    data = programs.pack_results(measurements)
    expected = make_bytes('\n        000 00\n        001 01\n        010 10\n        011 11\n        100 00\n        101 01\n        110 10\n\n        000 00 -- padding\n    ')
    assert data == expected