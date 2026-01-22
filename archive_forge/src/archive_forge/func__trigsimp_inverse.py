from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
def _trigsimp_inverse(rv):

    def check_args(x, y):
        try:
            return x.args[0] == y.args[0]
        except IndexError:
            return False

    def f(rv):
        g = getattr(rv, 'inverse', None)
        if g is not None and isinstance(rv.args[0], g()) and isinstance(g()(1), TrigonometricFunction):
            return rv.args[0].args[0]
        if isinstance(rv, atan2):
            y, x = rv.args
            if _coeff_isneg(y):
                return -f(atan2(-y, x))
            elif _coeff_isneg(x):
                return S.Pi - f(atan2(y, -x))
            if check_args(x, y):
                if isinstance(y, sin) and isinstance(x, cos):
                    return x.args[0]
                if isinstance(y, cos) and isinstance(x, sin):
                    return S.Pi / 2 - x.args[0]
        return rv
    return bottom_up(rv, f)