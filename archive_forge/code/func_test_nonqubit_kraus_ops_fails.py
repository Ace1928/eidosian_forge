import cirq
import numpy as np
import pytest
def test_nonqubit_kraus_ops_fails():
    ops = [np.array([[1, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 1, 0]])]
    with pytest.raises(ValueError, match='Input Kraus ops'):
        _ = cirq.KrausChannel(kraus_ops=ops, key='m')