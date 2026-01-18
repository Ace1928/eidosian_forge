import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_polynomial_weighted_estimation():
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    tform_w = estimate_transform('polynomial', SRC, DST, order=10, weights=np.ones(SRC.shape[0]))
    assert_almost_equal(tform.params, tform_w.params)
    point_weights = np.ones(SRC.shape[0] + 1)
    point_weights[0] = 1e-15
    tform1 = estimate_transform('polynomial', SRC, DST, order=10)
    tform2 = estimate_transform('polynomial', SRC[np.arange(-1, SRC.shape[0]), :], DST[np.arange(-1, SRC.shape[0]), :], order=10, weights=point_weights)
    assert_almost_equal(tform1.params, tform2.params, decimal=4)