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
def TR13(rv):
    """Change products of ``tan`` or ``cot``.

    Examples
    ========

    >>> from sympy.simplify.fu import TR13
    >>> from sympy import tan, cot
    >>> TR13(tan(3)*tan(2))
    -tan(2)/tan(5) - tan(3)/tan(5) + 1
    >>> TR13(cot(3)*cot(2))
    cot(2)*cot(5) + 1 + cot(3)*cot(5)
    """

    def f(rv):
        if not rv.is_Mul:
            return rv
        args = {tan: [], cot: [], None: []}
        for a in ordered(Mul.make_args(rv)):
            if a.func in (tan, cot):
                args[type(a)].append(a.args[0])
            else:
                args[None].append(a)
        t = args[tan]
        c = args[cot]
        if len(t) < 2 and len(c) < 2:
            return rv
        args = args[None]
        while len(t) > 1:
            t1 = t.pop()
            t2 = t.pop()
            args.append(1 - (tan(t1) / tan(t1 + t2) + tan(t2) / tan(t1 + t2)))
        if t:
            args.append(tan(t.pop()))
        while len(c) > 1:
            t1 = c.pop()
            t2 = c.pop()
            args.append(1 + cot(t1) * cot(t1 + t2) + cot(t2) * cot(t1 + t2))
        if c:
            args.append(cot(c.pop()))
        return Mul(*args)
    return bottom_up(rv, f)