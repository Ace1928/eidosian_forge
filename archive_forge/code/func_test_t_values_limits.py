import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('t', [np.linspace(-20, 20, 300), np.linspace(-0.0001, 0.0001, 17)])
def test_t_values_limits(self, t):
    with pytest.raises(ValueError, match='interpolation parameter'):
        _ = geometric_slerp(start=np.array([1, 0]), end=np.array([0, 1]), t=t)