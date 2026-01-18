import cirq
import numpy as np
import pytest
def test_nonqubit_mixture_fails():
    mix = [(0.5, np.array([[1, 0, 0], [0, 1, 0]])), (0.5, np.array([[0, 1, 0], [1, 0, 0]]))]
    with pytest.raises(ValueError, match='Input mixture'):
        _ = cirq.MixedUnitaryChannel(mixture=mix, key='m')