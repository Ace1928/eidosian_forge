from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def test_wrapped_func_no_args_no_kwargs():
    """
    Wrap a function that takes only positional-or-keyword parameters.
    """

    def f_auto_unc(x, y):
        return 2 * x + umath.sin(y)

    def f(x, y):
        assert not isinstance(x, uncert_core.UFloat)
        assert not isinstance(y, uncert_core.UFloat)
        return f_auto_unc(x, y)
    x = uncert_core.ufloat(1, 0.1)
    y = uncert_core.ufloat(10, 2)
    f_wrapped = uncert_core.wrap(f)
    assert ufloats_close(f_auto_unc(x, y), f_wrapped(x, y))
    assert ufloats_close(f_auto_unc(y=y, x=x), f_wrapped(y=y, x=x))
    f_wrapped = uncert_core.wrap(f, [None])
    assert ufloats_close(f_auto_unc(x, y), f_wrapped(x, y))
    assert ufloats_close(f_auto_unc(y=y, x=x), f_wrapped(y=y, x=x))
    f_wrapped = uncert_core.wrap(f, [lambda x, y: 2, lambda x, y: math.cos(y)])
    assert ufloats_close(f_auto_unc(x, y), f_wrapped(x, y))
    assert ufloats_close(f_auto_unc(y=y, x=x), f_wrapped(y=y, x=x))
    f_wrapped = uncert_core.wrap(f, [lambda x, y: 2])
    assert ufloats_close(f_auto_unc(x, y), f_wrapped(x, y))
    assert ufloats_close(f_auto_unc(y=y, x=x), f_wrapped(y=y, x=x))