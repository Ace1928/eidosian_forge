from collections import defaultdict
from sympy.core import sympify, S, Mul, Derivative, Pow
from sympy.core.add import _unevaluated_Add, Add
from sympy.core.assumptions import assumptions
from sympy.core.exprtools import Factors, gcd_terms
from sympy.core.function import _mexpand, expand_mul, expand_power_base
from sympy.core.mul import _keep_coeff, _unevaluated_Mul, _mulsort
from sympy.core.numbers import Rational, zoo, nan
from sympy.core.parameters import global_parameters
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Wild, symbols
from sympy.functions import exp, sqrt, log
from sympy.functions.elementary.complexes import Abs
from sympy.polys import gcd
from sympy.simplify.sqrtdenest import sqrtdenest
from sympy.utilities.iterables import iterable, sift
def _split_gcd(*a):
    """
    Split the list of integers ``a`` into a list of integers, ``a1`` having
    ``g = gcd(a1)``, and a list ``a2`` whose elements are not divisible by
    ``g``.  Returns ``g, a1, a2``.

    Examples
    ========

    >>> from sympy.simplify.radsimp import _split_gcd
    >>> _split_gcd(55, 35, 22, 14, 77, 10)
    (5, [55, 35, 10], [22, 14, 77])
    """
    g = a[0]
    b1 = [g]
    b2 = []
    for x in a[1:]:
        g1 = gcd(g, x)
        if g1 == 1:
            b2.append(x)
        else:
            g = g1
            b1.append(x)
    return (g, b1, b2)