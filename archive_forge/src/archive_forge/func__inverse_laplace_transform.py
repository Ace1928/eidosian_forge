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
def _inverse_laplace_transform(fn, s_, t_, plane, simplify=True, dorational=True):
    """
    Front-end function of the inverse Laplace transform. It tries to apply all
    known rules recursively.  If everything else fails, it tries to integrate.
    """
    terms = Add.make_args(fn)
    terms_t = []
    conditions = []
    debugf('[ILT _i_l_t] (%s, %s, %s)', (fn, s_, t_))
    for term in terms:
        k, f = term.as_independent(s_, as_Add=False)
        if dorational and term.is_rational_function(s_) and ((r := _inverse_laplace_rational(f, s_, t_, plane, simplify)) is not None):
            pass
        elif (r := _inverse_laplace_apply_simple_rules(f, s_, t_)) is not None:
            pass
        elif (r := _inverse_laplace_expand(f, s_, t_, plane)) is not None:
            pass
        elif (r := _inverse_laplace_apply_prog_rules(f, s_, t_, plane)) is not None:
            pass
        elif any((undef.has(s_) for undef in f.atoms(AppliedUndef))):
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        elif (r := _inverse_laplace_transform_integration(f, s_, t_, plane, simplify=simplify)) is not None:
            pass
        else:
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        ri_, ci_ = r
        terms_t.append(k * ri_)
        conditions.append(ci_)
    result = Add(*terms_t)
    if simplify:
        result = result.simplify(doit=False)
    condition = And(*conditions)
    return (result, condition)