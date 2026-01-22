from sympy.core import Add, Expr, Mul, S, sympify
from sympy.core.function import _mexpand, count_ops, expand_mul
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy
from sympy.functions import root, sign, sqrt
from sympy.polys import Poly, PolynomialError
def _sqrt_ratcomb(cs, args):
    """Denest rational combinations of radicals.

    Based on section 5 of [1].

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.simplify.sqrtdenest import sqrtdenest
    >>> z = sqrt(1+sqrt(3)) + sqrt(3+3*sqrt(3)) - sqrt(10+6*sqrt(3))
    >>> sqrtdenest(z)
    0
    """
    from sympy.simplify.radsimp import radsimp

    def find(a):
        n = len(a)
        for i in range(n - 1):
            for j in range(i + 1, n):
                s1 = a[i].base
                s2 = a[j].base
                p = _mexpand(s1 * s2)
                s = sqrtdenest(sqrt(p))
                if s != sqrt(p):
                    return (s, i, j)
    indices = find(args)
    if indices is None:
        return Add(*[c * arg for c, arg in zip(cs, args)])
    s, i1, i2 = indices
    c2 = cs.pop(i2)
    args.pop(i2)
    a1 = args[i1]
    cs[i1] += radsimp(c2 * s / a1.base)
    return _sqrt_ratcomb(cs, args)