import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_raises_for_non_hermitian_pauli():
    with pytest.raises(ValueError, match='hermitian'):
        cirq.PauliSumExponential(cirq.X(q0) + 1j * cirq.Z(q1), np.pi / 2)