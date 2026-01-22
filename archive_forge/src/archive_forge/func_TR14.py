from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
def TR14(rv, first=True):
    """Convert factored powers of sin and cos identities into simpler
    expressions.

    Examples
    ========

    >>> from sympy.simplify.fu import TR14
    >>> from sympy.abc import x, y
    >>> from sympy import cos, sin
    >>> TR14((cos(x) - 1)*(cos(x) + 1))
    -sin(x)**2
    >>> TR14((sin(x) - 1)*(sin(x) + 1))
    -cos(x)**2
    >>> p1 = (cos(x) + 1)*(cos(x) - 1)
    >>> p2 = (cos(y) - 1)*2*(cos(y) + 1)
    >>> p3 = (3*(cos(y) - 1))*(3*(cos(y) + 1))
    >>> TR14(p1*p2*p3*(x - 1))
    -18*(x - 1)*sin(x)**2*sin(y)**4

    """

    def f(rv):
        if not rv.is_Mul:
            return rv
        if first:
            n, d = rv.as_numer_denom()
            if d is not S.One:
                newn = TR14(n, first=False)
                newd = TR14(d, first=False)
                if newn != n or newd != d:
                    rv = newn / newd
                return rv
        other = []
        process = []
        for a in rv.args:
            if a.is_Pow:
                b, e = a.as_base_exp()
                if not (e.is_integer or b.is_positive):
                    other.append(a)
                    continue
                a = b
            else:
                e = S.One
            m = as_f_sign_1(a)
            if not m or m[1].func not in (cos, sin):
                if e is S.One:
                    other.append(a)
                else:
                    other.append(a ** e)
                continue
            g, f, si = m
            process.append((g, e.is_Number, e, f, si, a))
        process = list(ordered(process))
        nother = len(other)
        keys = g, t, e, f, si, a = list(range(6))
        while process:
            A = process.pop(0)
            if process:
                B = process[0]
                if A[e].is_Number and B[e].is_Number:
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = min(A[e], B[e])
                            if B[e] != take:
                                rem = [B[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            elif A[e] != take:
                                rem = [A[i] for i in keys]
                                rem[e] -= take
                                process.insert(0, rem)
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g] * B[g] * t(A[f].args[0]) ** 2) ** take)
                            continue
                elif A[e] == B[e]:
                    if A[f] == B[f]:
                        if A[si] != B[si]:
                            B = process.pop(0)
                            take = A[e]
                            if isinstance(A[f], cos):
                                t = sin
                            else:
                                t = cos
                            other.append((-A[g] * B[g] * t(A[f].args[0]) ** 2) ** take)
                            continue
            other.append(A[a] ** A[e])
        if len(other) != nother:
            rv = Mul(*other)
        return rv
    return bottom_up(rv, f)