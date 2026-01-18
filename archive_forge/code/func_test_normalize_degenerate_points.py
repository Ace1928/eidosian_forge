import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_normalize_degenerate_points():
    """Return nan matrix *of appropriate size* when point is repeated."""
    pts = np.array([[73.42834308, 94.2977623]] * 3)
    mat, pts_tf = _center_and_normalize_points(pts)
    assert np.all(np.isnan(mat))
    assert np.all(np.isnan(pts_tf))
    assert mat.shape == (3, 3)
    assert pts_tf.shape == pts.shape