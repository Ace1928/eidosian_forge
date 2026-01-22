from functools import reduce
from sympy.core.add import Add
from sympy.core.exprtools import Factors
from sympy.core.function import expand_mul, expand_multinomial, _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi, _illegal)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin, tan
from sympy.ntheory.factor_ import divisors
from sympy.utilities.iterables import subsets
from sympy.polys.domains import ZZ, QQ, FractionField
from sympy.polys.orthopolys import dup_chebyshevt
from sympy.polys.polyerrors import (
from sympy.polys.polytools import (
from sympy.polys.polyutils import dict_from_expr, expr_from_dict
from sympy.polys.ring_series import rs_compose_add
from sympy.polys.rings import ring
from sympy.polys.rootoftools import CRootOf
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import (
def _minpoly_tan(ex, x):
    """
    Returns the minimal polynomial of ``tan(ex)``
    see https://github.com/sympy/sympy/issues/21430
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            c = c * 2
            n = int(c.q)
            a = n if c.p % 2 == 0 else 1
            terms = []
            for k in range((c.p + 1) % 2, n + 1, 2):
                terms.append(a * x ** k)
                a = -(a * (n - k - 1) * (n - k)) // ((k + 1) * (k + 2))
            r = Add(*terms)
            _, factors = factor_list(r)
            res = _choose_factor(factors, x, ex)
            return res
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)