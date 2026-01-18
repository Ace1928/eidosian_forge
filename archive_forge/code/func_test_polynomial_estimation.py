import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_polynomial_estimation():
    tform = estimate_transform('polynomial', SRC, DST, order=10)
    assert_almost_equal(tform(SRC), DST, 6)
    tform2 = PolynomialTransform()
    assert tform2.estimate(SRC, DST, order=10)
    assert_almost_equal(tform2.params, tform.params)