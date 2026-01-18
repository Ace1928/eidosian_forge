import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('tol', [-5e-06, -7e-10])
def test_tol_sign(self, tol):
    _ = geometric_slerp(start=np.array([1, 0]), end=np.array([0, 1]), t=np.linspace(0, 1, 5), tol=tol)