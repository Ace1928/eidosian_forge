from math import isclose
from sympy.core.numbers import I
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import (Abs, arg)
from sympy.functions.elementary.exponential import log
from sympy.abc import s, p, a
from sympy.external import import_module
from sympy.physics.control.control_plots import \
from sympy.physics.control.lti import (TransferFunction,
from sympy.testing.pytest import raises, skip
def _trim_tuple(a, b):
    a, b = _to_tuple(a, b)
    return (tuple(a[0:2] + a[len(a) // 2:len(a) // 2 + 1] + a[-2:]), tuple(b[0:2] + b[len(b) // 2:len(b) // 2 + 1] + b[-2:]))