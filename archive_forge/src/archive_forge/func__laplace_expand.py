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
def _laplace_expand(f, t, s, doit=True, **hints):
    """
    This function tries to expand its argument with successively stronger
    methods: first it will expand on the top level, then it will expand any
    multiplications in depth, then it will try all avilable expansion methods,
    and finally it will try to expand trigonometric functions.

    If it can expand, it will then compute the Laplace transform of the
    expanded term.
    """
    if f.is_Add:
        return None
    r = expand(f, deep=False)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand_mul(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    if r != f:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand(expand_trig(f))
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    return None