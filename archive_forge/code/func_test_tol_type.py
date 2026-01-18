import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('tol', [5, '7', [5, 6, 7], np.array(9.0)])
def test_tol_type(self, tol):
    with pytest.raises(ValueError, match='must be a float'):
        _ = geometric_slerp(start=np.array([1, 0]), end=np.array([0, 1]), t=np.linspace(0, 1, 5), tol=tol)