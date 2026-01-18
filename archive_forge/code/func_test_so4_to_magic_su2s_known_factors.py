import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('a,b', [(cirq.testing.random_special_unitary(2), cirq.testing.random_special_unitary(2)) for _ in range(10)])
def test_so4_to_magic_su2s_known_factors(a, b):
    m = recompose_so4(a, b)
    a2, b2 = cirq.so4_to_magic_su2s(m)
    m2 = recompose_so4(a2, b2)
    assert np.allclose(m2, m)
    if np.linalg.norm(a + a2) > np.linalg.norm(a - a2):
        assert np.allclose(a2, a)
        assert np.allclose(b2, b)
    else:
        assert np.allclose(a2, -a)
        assert np.allclose(b2, -b)