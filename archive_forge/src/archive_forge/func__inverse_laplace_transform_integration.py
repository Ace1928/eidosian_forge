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
def _inverse_laplace_transform_integration(F, s, t_, plane, simplify=True):
    """ The backend function for inverse Laplace transforms. """
    from sympy.integrals.meijerint import meijerint_inversion, _get_coeff_exp
    from sympy.integrals.transforms import inverse_mellin_transform
    t = Dummy('t', real=True)

    def pw_simp(*args):
        """ Simplify a piecewise expression from hyperexpand. """
        if len(args) != 3:
            return Piecewise(*args)
        arg = args[2].args[0].argument
        coeff, exponent = _get_coeff_exp(arg, t)
        e1 = args[0].args[0]
        e2 = args[1].args[0]
        return Heaviside(1 / Abs(coeff) - t ** exponent) * e1 + Heaviside(t ** exponent - 1 / Abs(coeff)) * e2
    if F.is_rational_function(s):
        F = F.apart(s)
    if F.is_Add:
        f = Add(*[_inverse_laplace_transform_integration(X, s, t, plane, simplify) for X in F.args])
        return (_simplify(f.subs(t, t_), simplify), True)
    try:
        f, cond = inverse_mellin_transform(F, s, exp(-t), (None, S.Infinity), needeval=True, noconds=False)
    except IntegralTransformError:
        f = None
    if f is None:
        f = meijerint_inversion(F, s, t)
        if f is None:
            return None
        if f.is_Piecewise:
            f, cond = f.args[0]
            if f.has(Integral):
                return None
        else:
            cond = S.true
        f = f.replace(Piecewise, pw_simp)
    if f.is_Piecewise:
        return (f.subs(t, t_), cond)
    u = Dummy('u')

    def simp_heaviside(arg, H0=S.Half):
        a = arg.subs(exp(-t), u)
        if a.has(t):
            return Heaviside(arg, H0)
        from sympy.solvers.inequalities import _solve_inequality
        rel = _solve_inequality(a > 0, u)
        if rel.lts == u:
            k = log(rel.gts)
            return Heaviside(t + k, H0)
        else:
            k = log(rel.lts)
            return Heaviside(-(t + k), H0)
    f = f.replace(Heaviside, simp_heaviside)

    def simp_exp(arg):
        return expand_complex(exp(arg))
    f = f.replace(exp, simp_exp)
    return (_simplify(f.subs(t, t_), simplify), cond)