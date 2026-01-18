import warnings
from scipy._lib import _pep440
import numpy as np
from numpy.testing import (assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from numpy import array, spacing, sin, pi, sort, sqrt
from scipy.signal import (argrelextrema, BadCoefficients, bessel, besselap, bilinear,
from scipy.signal._filter_design import (_cplxreal, _cplxpair, _norm_factor,
def test_sos_consistency():
    design_funcs = [(bessel, (0.1,)), (butter, (0.1,)), (cheby1, (45.0, 0.1)), (cheby2, (0.087, 0.1)), (ellip, (0.087, 45, 0.1))]
    for func, args in design_funcs:
        name = func.__name__
        b, a = func(2, *args, output='ba')
        sos = func(2, *args, output='sos')
        assert_allclose(sos, [np.hstack((b, a))], err_msg='%s(2,...)' % name)
        zpk = func(3, *args, output='zpk')
        sos = func(3, *args, output='sos')
        assert_allclose(sos, zpk2sos(*zpk), err_msg='%s(3,...)' % name)
        zpk = func(4, *args, output='zpk')
        sos = func(4, *args, output='sos')
        assert_allclose(sos, zpk2sos(*zpk), err_msg='%s(4,...)' % name)