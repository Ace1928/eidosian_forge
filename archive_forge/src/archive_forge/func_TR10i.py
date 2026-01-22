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
def TR10i(rv):
    """Sum of products to function of sum.

    Examples
    ========

    >>> from sympy.simplify.fu import TR10i
    >>> from sympy import cos, sin, sqrt
    >>> from sympy.abc import x

    >>> TR10i(cos(1)*cos(3) + sin(1)*sin(3))
    cos(2)
    >>> TR10i(cos(1)*sin(3) + sin(1)*cos(3) + cos(3))
    cos(3) + sin(4)
    >>> TR10i(sqrt(2)*cos(x)*x + sqrt(6)*sin(x)*x)
    2*sqrt(2)*x*sin(x + pi/6)

    """
    global _ROOT2, _ROOT3, _invROOT3
    if _ROOT2 is None:
        _roots()

    def f(rv):
        if not rv.is_Add:
            return rv

        def do(rv, first=True):
            if not rv.is_Add:
                return rv
            args = list(ordered(rv.args))
            if len(args) != 2:
                hit = False
                for i in range(len(args)):
                    ai = args[i]
                    if ai is None:
                        continue
                    for j in range(i + 1, len(args)):
                        aj = args[j]
                        if aj is None:
                            continue
                        was = ai + aj
                        new = do(was)
                        if new != was:
                            args[i] = new
                            args[j] = None
                            hit = True
                            break
                if hit:
                    rv = Add(*[_f for _f in args if _f])
                    if rv.is_Add:
                        rv = do(rv)
                return rv
            split = trig_split(*args, two=True)
            if not split:
                return rv
            gcd, n1, n2, a, b, same = split
            if same:
                gcd = n1 * gcd
                if n1 == n2:
                    return gcd * cos(a - b)
                return gcd * cos(a + b)
            else:
                gcd = n1 * gcd
                if n1 == n2:
                    return gcd * sin(a + b)
                return gcd * sin(b - a)
        rv = process_common_addends(rv, do, lambda x: tuple(ordered(x.free_symbols)))
        while rv.is_Add:
            byrad = defaultdict(list)
            for a in rv.args:
                hit = 0
                if a.is_Mul:
                    for ai in a.args:
                        if ai.is_Pow and ai.exp is S.Half and ai.base.is_Integer:
                            byrad[ai].append(a)
                            hit = 1
                            break
                if not hit:
                    byrad[S.One].append(a)
            args = []
            for a in byrad:
                for b in [_ROOT3 * a, _invROOT3]:
                    if b in byrad:
                        for i in range(len(byrad[a])):
                            if byrad[a][i] is None:
                                continue
                            for j in range(len(byrad[b])):
                                if byrad[b][j] is None:
                                    continue
                                was = Add(byrad[a][i] + byrad[b][j])
                                new = do(was)
                                if new != was:
                                    args.append(new)
                                    byrad[a][i] = None
                                    byrad[b][j] = None
                                    break
            if args:
                rv = Add(*args + [Add(*[_f for _f in v if _f]) for v in byrad.values()])
            else:
                rv = do(rv)
                break
        return rv
    return bottom_up(rv, f)