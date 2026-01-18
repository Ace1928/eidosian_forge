import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_norm_factor(self):
    mpmath_values = {1: 1, 2: 1.3616541287161306, 3: 1.7556723686812106, 4: 2.113917674904216, 5: 2.427410702152628, 6: 2.703395061202922, 7: 2.9517221470387227, 8: 3.1796172375106515, 9: 3.39169313891166, 10: 3.5909805945691633, 11: 3.77960741643962, 12: 3.959150821144285, 13: 4.130825499383536, 14: 4.295593409533637, 15: 4.454233021624377, 16: 4.607385465472648, 17: 4.755586548961148, 18: 4.899289677284488, 19: 5.038882681488207, 20: 5.174700441742708, 21: 5.307034531360917, 22: 5.436140703250036, 23: 5.562244783787878, 24: 5.685547371295963, 25: 5.806227623775419, 50: 8.268963160013227, 51: 8.352374541546013}
    for N in mpmath_values:
        z, p, k = besselap(N, 'delay')
        assert_allclose(mpmath_values[N], _norm_factor(p, k), rtol=1e-13)