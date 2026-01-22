from sympy.core import S, pi, I
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.function import (
from sympy.core.mul import Mul, prod
from sympy.core.relational import _canonical, Ge, Gt, Lt, Unequality, Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.complexes import (
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, asinh
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.special.gamma_functions import digamma, gamma, lowergamma
from sympy.integrals import integrate, Integral
from sympy.integrals.transforms import (
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.matrices.matrices import MatrixBase
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.utilities.exceptions import (
from sympy.utilities.misc import debug, debugf
def _simplifyconds(expr, s, a):
    """
    Naively simplify some conditions occurring in ``expr``,
    given that `\\operatorname{Re}(s) > a`.

    Examples
    ========

    >>> from sympy.integrals.laplace import _simplifyconds
    >>> from sympy.abc import x
    >>> from sympy import sympify as S
    >>> _simplifyconds(abs(x**2) < 1, x, 1)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 2)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 0)
    Abs(x**2) < 1
    >>> _simplifyconds(abs(1/x**2) < 1, x, 1)
    True
    >>> _simplifyconds(S(1) < abs(x), x, 1)
    True
    >>> _simplifyconds(S(1) < abs(1/x), x, 1)
    False

    >>> from sympy import Ne
    >>> _simplifyconds(Ne(1, x**3), x, 1)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 2)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 0)
    Ne(1, x**3)
    """

    def power(ex):
        if ex == s:
            return 1
        if ex.is_Pow and ex.base == s:
            return ex.exp
        return None

    def bigger(ex1, ex2):
        """ Return True only if |ex1| > |ex2|, False only if |ex1| < |ex2|.
            Else return None. """
        if ex1.has(s) and ex2.has(s):
            return None
        if isinstance(ex1, Abs):
            ex1 = ex1.args[0]
        if isinstance(ex2, Abs):
            ex2 = ex2.args[0]
        if ex1.has(s):
            return bigger(1 / ex2, 1 / ex1)
        n = power(ex2)
        if n is None:
            return None
        try:
            if n > 0 and (Abs(ex1) <= Abs(a) ** n) == S.true:
                return False
            if n < 0 and (Abs(ex1) >= Abs(a) ** n) == S.true:
                return True
        except TypeError:
            pass

    def replie(x, y):
        """ simplify x < y """
        if not (x.is_positive or isinstance(x, Abs)) or not (y.is_positive or isinstance(y, Abs)):
            return x < y
        r = bigger(x, y)
        if r is not None:
            return not r
        return x < y

    def replue(x, y):
        b = bigger(x, y)
        if b in (True, False):
            return True
        return Unequality(x, y)

    def repl(ex, *args):
        if ex in (True, False):
            return bool(ex)
        return ex.replace(*args)
    from sympy.simplify.radsimp import collect_abs
    expr = collect_abs(expr)
    expr = repl(expr, Lt, replie)
    expr = repl(expr, Gt, lambda x, y: replie(y, x))
    expr = repl(expr, Unequality, replue)
    return S(expr)