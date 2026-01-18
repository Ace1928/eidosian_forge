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
def test_wrapped_func_args_no_kwargs():
    """
    Wrap a function that takes only positional-or-keyword and
    var-positional parameters.
    """

    def f_auto_unc(x, y, *args):
        return 2 * x + umath.sin(y) + 3 * args[1]

    def f(x, y, *args):
        assert not any((isinstance(value, uncert_core.UFloat) for value in [x, y] + list(args)))
        return f_auto_unc(x, y, *args)
    x = uncert_core.ufloat(1, 0.1)
    y = uncert_core.ufloat(10, 2)
    s = 'string arg'
    z = uncert_core.ufloat(100, 3)
    args = [s, z, s]
    f_wrapped = uncert_core.wrap(f)
    assert ufloats_close(f_auto_unc(x, y, *args), f_wrapped(x, y, *args))
    f_wrapped = uncert_core.wrap(f, [None])
    assert ufloats_close(f_auto_unc(x, y, *args), f_wrapped(x, y, *args))
    f_wrapped = uncert_core.wrap(f, [lambda x, y, *args: 2, lambda x, y, *args: math.cos(y), None, lambda x, y, *args: 3])
    assert ufloats_close(f_auto_unc(x, y, *args), f_wrapped(x, y, *args))
    f_wrapped = uncert_core.wrap(f, [lambda x, y, *args: 2])
    assert ufloats_close(f_auto_unc(x, y, *args), f_wrapped(x, y, *args))