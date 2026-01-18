from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_pdf_logpdf_validation():
    rng = np.random.default_rng(64202298293133848336925499069837723291)
    xn = rng.standard_normal((2, 10))
    gkde = stats.gaussian_kde(xn)
    xs = rng.standard_normal((3, 10))
    msg = 'points have dimension 3, dataset has dimension 2'
    with pytest.raises(ValueError, match=msg):
        gkde.logpdf(xs)