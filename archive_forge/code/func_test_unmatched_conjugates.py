import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_unmatched_conjugates(self):
    assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 2j])
    assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 2j, 1 - 3j])
    assert_raises(ValueError, _cplxreal, [1 + 3j, 1 - 3j, 1 + 3j])
    assert_raises(ValueError, _cplxreal, [1 + 3j])
    assert_raises(ValueError, _cplxreal, [1 - 3j])