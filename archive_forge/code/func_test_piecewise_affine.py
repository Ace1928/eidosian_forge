import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_piecewise_affine():
    tform = PiecewiseAffineTransform()
    assert tform.estimate(SRC, DST)
    assert_almost_equal(tform(SRC), DST)
    assert_almost_equal(tform.inverse(DST), SRC)