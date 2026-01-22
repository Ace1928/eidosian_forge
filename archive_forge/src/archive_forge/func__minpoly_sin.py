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
def _minpoly_sin(ex, x):
    """
    Returns the minimal polynomial of ``sin(ex)``
    see https://mathworld.wolfram.com/TrigonometryAngles.html
    """
    c, a = ex.args[0].as_coeff_Mul()
    if a is pi:
        if c.is_rational:
            n = c.q
            q = sympify(n)
            if q.is_prime:
                a = dup_chebyshevt(n, ZZ)
                return Add(*[x ** (n - i - 1) * a[i] for i in range(n)])
            if c.p == 1:
                if q == 9:
                    return 64 * x ** 6 - 96 * x ** 4 + 36 * x ** 2 - 3
            if n % 2 == 1:
                a = dup_chebyshevt(n, ZZ)
                a = [x ** (n - i) * a[i] for i in range(n + 1)]
                r = Add(*a)
                _, factors = factor_list(r)
                res = _choose_factor(factors, x, ex)
                return res
            expr = ((1 - cos(2 * c * pi)) / 2) ** S.Half
            res = _minpoly_compose(expr, x, QQ)
            return res
    raise NotAlgebraic('%s does not seem to be an algebraic element' % ex)