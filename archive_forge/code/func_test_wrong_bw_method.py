from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_wrong_bw_method(self):
    """Test the error message that should be called when bw is invalid."""
    np.random.seed(8765678)
    n_basesample = 50
    data = np.random.randn(n_basesample)
    with pytest.raises(ValueError):
        mlab.GaussianKDE(data, bw_method='invalid')