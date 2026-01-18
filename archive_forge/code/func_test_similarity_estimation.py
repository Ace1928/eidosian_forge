import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_similarity_estimation():
    tform = estimate_transform('similarity', SRC[:2, :], DST[:2, :])
    assert_almost_equal(tform(SRC[:2, :]), DST[:2, :])
    assert_almost_equal(tform.params[0, 0], tform.params[1, 1])
    assert_almost_equal(tform.params[0, 1], -tform.params[1, 0])
    tform2 = estimate_transform('similarity', SRC, DST)
    assert_almost_equal(tform2.inverse(tform2(SRC)), SRC)
    assert_almost_equal(tform2.params[0, 0], tform2.params[1, 1])
    assert_almost_equal(tform2.params[0, 1], -tform2.params[1, 0])
    tform3 = SimilarityTransform()
    assert tform3.estimate(SRC, DST)
    assert_almost_equal(tform3.params, tform2.params)