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
def TR4(rv):
    """Identify values of special angles.

        a=  0   pi/6        pi/4        pi/3        pi/2
    ----------------------------------------------------
    sin(a)  0   1/2         sqrt(2)/2   sqrt(3)/2   1
    cos(a)  1   sqrt(3)/2   sqrt(2)/2   1/2         0
    tan(a)  0   sqt(3)/3    1           sqrt(3)     --

    Examples
    ========

    >>> from sympy import pi
    >>> from sympy import cos, sin, tan, cot
    >>> for s in (0, pi/6, pi/4, pi/3, pi/2):
    ...    print('%s %s %s %s' % (cos(s), sin(s), tan(s), cot(s)))
    ...
    1 0 0 zoo
    sqrt(3)/2 1/2 sqrt(3)/3 sqrt(3)
    sqrt(2)/2 sqrt(2)/2 1 1
    1/2 sqrt(3)/2 sqrt(3) sqrt(3)/3
    0 1 zoo 0
    """
    return rv