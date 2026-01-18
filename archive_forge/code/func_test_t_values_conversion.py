import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp
@pytest.mark.parametrize('t', ['15, 5, 7'])
def test_t_values_conversion(self, t):
    with pytest.raises(ValueError):
        _ = geometric_slerp(start=np.array([1]), end=np.array([0]), t=t)