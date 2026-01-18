import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_wn1_ge_wn0(self):
    with pytest.raises(ValueError, match='Wn\\[0\\] must be less than Wn\\[1\\]'):
        iirfilter(2, [0.5, 0.5])
    with pytest.raises(ValueError, match='Wn\\[0\\] must be less than Wn\\[1\\]'):
        iirfilter(2, [0.6, 0.5])