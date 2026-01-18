import numpy as np
from skimage.util import regular_grid
from skimage._shared.testing import assert_equal
def test_regular_grid_2d_32():
    ar = np.zeros((20, 40))
    g = regular_grid(ar.shape, 32)
    assert_equal(g, [slice(2.0, None, 5.0), slice(2.0, None, 5.0)])
    ar[g] = 1
    assert_equal(ar.sum(), 32)