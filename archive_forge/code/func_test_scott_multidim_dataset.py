from numpy.testing import (assert_allclose, assert_almost_equal,
import numpy as np
import pytest
from matplotlib import mlab
def test_scott_multidim_dataset(self):
    """Test scott's output for a multi-dimensional array."""
    x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    with pytest.raises(np.linalg.LinAlgError):
        mlab.GaussianKDE(x1, 'scott')