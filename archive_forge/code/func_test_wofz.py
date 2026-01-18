import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
def test_wofz(self):
    z = [complex(624.2, -0.26123), complex(-0.4, 3.0), complex(0.6, 2.0), complex(-1.0, 1.0), complex(-1.0, -9.0), complex(-1.0, 9.0), complex(-2.34545e-08, 1.1234), complex(-3.0, 5.1), complex(-53, 30.1), complex(0.0, 0.12345), complex(11, 1), complex(-22, -2), complex(9, -28), complex(21, -33), complex(100000.0, 100000.0), complex(100000000000000.0, 100000000000000.0)]
    w = [complex(-3.7827024551898053e-07, 0.0009038612764331721), complex(0.1764906227004817, -0.021465505394684576), complex(0.2410250715772692, 0.060875796634280895), complex(0.3047442052569126, -0.20821893820283163), complex(7.317131068972378e+34, 8.321873499714403e+34), complex(0.06156985072363237, -0.00676005783716575), complex(0.3960793007699875, -5.593152259116645e-09), complex(0.08217199226739448, -0.0470129108764361), complex(0.0045724600035028165, -0.008049007914116918), complex(0.8746342859608053, 0.0), complex(0.004681901649654442, 0.05107355639013062), complex(-0.002319317520018762, -0.025460054739731557), complex(9.114633684056372e+304, 3.971018071452634e+305), complex(-4.49272078577156e+281, -2.8019591213423078e+281), complex(2.8209479178093053e-06, 2.8209479176682578e-06), complex(2.8209479177387813e-15, 2.8209479177387813e-15)]
    assert_func_equal(cephes.wofz, w, z, rtol=1e-13)