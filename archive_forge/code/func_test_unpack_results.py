import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_unpack_results():
    data = make_bytes('\n        000 00\n        001 01\n        010 10\n        011 11\n        100 00\n        101 01\n        110 10\n    ')
    assert len(data) == 5
    results = programs.unpack_results(data, 7, [('a', 3), ('b', 2)])
    assert 'a' in results
    assert results['a'].shape == (7, 3)
    assert results['a'].dtype == bool
    np.testing.assert_array_equal(results['a'], [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    assert 'b' in results
    assert results['b'].shape == (7, 2)
    assert results['b'].dtype == bool
    np.testing.assert_array_equal(results['b'], [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0]])