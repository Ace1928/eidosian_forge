import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('k', np.logspace(-10, -1, 10))
def test_numerical_stability_pi(self, k):
    angle = np.pi - k
    ts = np.linspace(0, 1, 100)
    P = np.array([1, 0, 0, 0])
    Q = np.array([np.cos(angle), np.sin(angle), 0, 0])
    with np.testing.suppress_warnings() as sup:
        sup.filter(UserWarning)
        result = geometric_slerp(P, Q, ts, 1e-18)
        norms = np.linalg.norm(result, axis=1)
        error = np.max(np.abs(norms - 1))
        assert error < 4e-15