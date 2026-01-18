from typing import Tuple as tTuple
from sympy.core import S, Add, Mul, sympify, Symbol, Dummy, Basic
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import (Function, Derivative, ArgumentIndexError,
from sympy.core.logic import fuzzy_not, fuzzy_or
from sympy.core.numbers import pi, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
def unbranched_argument(arg):
    """
    Returns periodic argument of arg with period as infinity.

    Examples
    ========

    >>> from sympy import exp_polar, unbranched_argument
    >>> from sympy import I, pi
    >>> unbranched_argument(exp_polar(15*I*pi))
    15*pi
    >>> unbranched_argument(exp_polar(7*I*pi))
    7*pi

    See also
    ========

    periodic_argument
    """
    return periodic_argument(arg, oo)