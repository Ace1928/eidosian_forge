import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
def test_result_addition_invalid():
    a = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), measurements={'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool), 'q1': np.array([[0], [0], [1]], dtype=bool)})
    b = cirq.ResultDict(params=cirq.ParamResolver({'bad': 1}), measurements={'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool), 'q1': np.array([[0], [0], [1]], dtype=bool)})
    c = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), measurements={'bad': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool), 'q1': np.array([[0], [0], [1]], dtype=bool)})
    d = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), measurements={'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool), 'q1': np.array([[0, 1], [0, 1], [1, 1]], dtype=bool)})
    e = cirq.ResultDict(params=cirq.ParamResolver({'ax': 1}), records={'q0': np.array([[0, 1], [1, 0], [0, 1]], dtype=bool), 'q1': np.array([[[0], [0]], [[0], [1]], [[1], [0]]], dtype=bool)})
    with pytest.raises(ValueError, match='different parameters'):
        _ = a + b
    with pytest.raises(ValueError, match='different measurement shapes'):
        _ = a + c
    with pytest.raises(ValueError, match='different measurement shapes'):
        _ = a + d
    with pytest.raises(ValueError, match='different measurement shapes'):
        _ = a + e
    with pytest.raises(TypeError):
        _ = a + 'junk'