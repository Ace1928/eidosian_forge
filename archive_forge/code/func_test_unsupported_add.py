import pickle
import numpy as np
import pytest
import cirq
def test_unsupported_add():
    with pytest.raises(TypeError, match='1'):
        _ = cirq.GridQubit(1, 1) + 1
    with pytest.raises(TypeError, match='(1,)'):
        _ = cirq.GridQubit(1, 1) + (1,)
    with pytest.raises(TypeError, match='(1, 2, 3)'):
        _ = cirq.GridQubit(1, 1) + (1, 2, 3)
    with pytest.raises(TypeError, match='(1, 2.0)'):
        _ = cirq.GridQubit(1, 1) + (1, 2.0)
    with pytest.raises(TypeError, match='1'):
        _ = cirq.GridQubit(1, 1) - 1
    with pytest.raises(TypeError, match='[1., 2.]'):
        _ = cirq.GridQubit(1, 1) + np.array([1.0, 2.0])
    with pytest.raises(TypeError, match='[1, 2, 3]'):
        _ = cirq.GridQubit(1, 1) + np.array([1, 2, 3], dtype=int)