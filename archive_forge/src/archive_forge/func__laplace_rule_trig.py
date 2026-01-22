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
def _laplace_rule_trig(fn, t_, s, doit=True, **hints):
    """
    This rule covers trigonometric factors by splitting everything into a
    sum of exponential functions and collecting complex conjugate poles and
    real symmetric poles.
    """
    t = Dummy('t', real=True)
    if not fn.has(sin, cos, sinh, cosh):
        return None
    debugf('_laplace_rule_trig: (%s, %s, %s)', (fn, t_, s))
    f, g = _laplace_trig_split(fn.subs(t_, t))
    debugf('    f = %s\n    g = %s', (f, g))
    xm, xn = _laplace_trig_expsum(f, t)
    debugf('    xm = %s\n    xn = %s', (xm, xn))
    if len(xn) > 0:
        debug('    --> xn is not empty; giving up.')
        return None
    if not g.has(t):
        r, p = _laplace_trig_ltex(xm, t, s)
        return (g * r, p, S.true)
    else:
        planes = []
        results = []
        G, G_plane, G_cond = _laplace_transform(g, t, s)
        for x1 in xm:
            results.append(x1['k'] * G.subs(s, s - x1['a']))
            planes.append(G_plane + re(x1['a']))
    return (Add(*results).subs(t, t_), Max(*planes), G_cond)