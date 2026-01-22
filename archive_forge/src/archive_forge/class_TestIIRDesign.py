import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
class TestIIRDesign:

    def test_exceptions(self):
        with pytest.raises(ValueError, match='the same shape'):
            iirdesign(0.2, [0.1, 0.3], 1, 40)
        with pytest.raises(ValueError, match='the same shape'):
            iirdesign(np.array([[0.3, 0.6], [0.3, 0.6]]), np.array([[0.4, 0.5], [0.4, 0.5]]), 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(-0.1, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0.1, 0, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0.1, -0.5, 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([-0.1, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, -0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [-0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0.1, 0], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0.1, -0.5], 1, 40)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(-0.1, 0.5, 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign(0.1, -0.5, 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([-0.1, 0.3], [0.1, 0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, -0.3], [0.1, 0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [-0.1, 0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be greater than 0'):
            iirdesign([0.1, 0.3], [0.1, -0.5], 1, 40, analog=True)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(1, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(1.1, 0.5, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(0.1, 1, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign(0.1, 1.5, 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([1, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([1.1, 0.3], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 1], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 1.1], [0.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [1.1, 0.5], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [0.1, 1], 1, 40)
        with pytest.raises(ValueError, match='must be less than 1'):
            iirdesign([0.1, 0.3], [0.1, 1.5], 1, 40)
        iirdesign(100, 500, 1, 40, fs=2000)
        iirdesign(500, 100, 1, 40, fs=2000)
        iirdesign([200, 400], [100, 500], 1, 40, fs=2000)
        iirdesign([100, 500], [200, 400], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(1000, 400, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(1100, 500, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(100, 1000, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign(100, 1100, 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([1000, 400], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([1100, 400], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 1000], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 1100], [100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [1000, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [1100, 500], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [100, 1000], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='must be less than fs/2'):
            iirdesign([200, 400], [100, 1100], 1, 40, fs=2000)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.1, 0.4], [0.5, 0.6], 1, 40)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.5, 0.6], [0.1, 0.4], 1, 40)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.3, 0.6], [0.4, 0.7], 1, 40)
        with pytest.raises(ValueError, match='strictly inside stopband'):
            iirdesign([0.4, 0.7], [0.3, 0.6], 1, 40)