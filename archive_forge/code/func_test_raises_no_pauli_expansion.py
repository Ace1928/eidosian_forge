import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('val', (NoMethod(), ReturnsNotImplemented(), HasQuditUnitary(), 123, np.eye(2), object(), cirq))
def test_raises_no_pauli_expansion(val):
    assert cirq.pauli_expansion(val, default=None) is None
    with pytest.raises(TypeError, match='No Pauli expansion'):
        cirq.pauli_expansion(val)