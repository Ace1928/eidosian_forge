import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_projective_weighted_estimation():
    tform = estimate_transform('projective', SRC[:4, :], DST[:4, :])
    tform_w = estimate_transform('projective', SRC[:4, :], DST[:4, :], np.ones(4))
    assert_almost_equal(tform.params, tform_w.params)
    tform = estimate_transform('projective', SRC, DST)
    tform_w = estimate_transform('projective', SRC, DST, np.ones(SRC.shape[0]))
    assert_almost_equal(tform.params, tform_w.params)
    point_weights = np.ones(SRC.shape[0] + 1)
    point_weights[0] = 1e-15
    tform1 = estimate_transform('projective', SRC, DST)
    tform2 = estimate_transform('projective', SRC[np.arange(-1, SRC.shape[0]), :], DST[np.arange(-1, SRC.shape[0]), :], point_weights)
    assert_almost_equal(tform1.params, tform2.params, decimal=3)