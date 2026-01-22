from math import prod
from collections import defaultdict
from typing import Tuple as tTuple
from sympy.core import S, Symbol, Add, Dummy
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import ArgumentIndexError, Function, expand_mul
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import E, I, pi, oo, Rational, Integer
from sympy.core.relational import Eq, is_le, is_gt
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import (binomial,
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.ntheory.primetest import isprime, is_square
from sympy.polys.appellseqs import bernoulli_poly, euler_poly, genocchi_poly
from sympy.utilities.enumerative import MultisetPartitionTraverser
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import multiset, multiset_derangements, iterable
from sympy.utilities.memoization import recurrence_memo
from sympy.utilities.misc import as_int
from mpmath import mp, workprec
from mpmath.libmp import ifib as _ifib
class tribonacci(Function):
    """
    Tribonacci numbers / Tribonacci polynomials

    The Tribonacci numbers are the integer sequence defined by the
    initial terms `T_0 = 0`, `T_1 = 1`, `T_2 = 1` and the three-term
    recurrence relation `T_n = T_{n-1} + T_{n-2} + T_{n-3}`.

    The Tribonacci polynomials are defined by `T_0(x) = 0`, `T_1(x) = 1`,
    `T_2(x) = x^2`, and `T_n(x) = x^2 T_{n-1}(x) + x T_{n-2}(x) + T_{n-3}(x)`
    for `n > 2`.  For all positive integers `n`, `T_n(1) = T_n`.

    * ``tribonacci(n)`` gives the `n^{th}` Tribonacci number, `T_n`
    * ``tribonacci(n, x)`` gives the `n^{th}` Tribonacci polynomial in `x`, `T_n(x)`

    Examples
    ========

    >>> from sympy import tribonacci, Symbol

    >>> [tribonacci(x) for x in range(11)]
    [0, 1, 1, 2, 4, 7, 13, 24, 44, 81, 149]
    >>> tribonacci(5, Symbol('t'))
    t**8 + 3*t**5 + 3*t**2

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Generalizations_of_Fibonacci_numbers#Tribonacci_numbers
    .. [2] https://mathworld.wolfram.com/TribonacciNumber.html
    .. [3] https://oeis.org/A000073

    """

    @staticmethod
    @recurrence_memo([S.Zero, S.One, S.One])
    def _trib(n, prev):
        return prev[-3] + prev[-2] + prev[-1]

    @staticmethod
    @recurrence_memo([S.Zero, S.One, _sym ** 2])
    def _tribpoly(n, prev):
        return (prev[-3] + _sym * prev[-2] + _sym ** 2 * prev[-1]).expand()

    @classmethod
    def eval(cls, n, sym=None):
        if n is S.Infinity:
            return S.Infinity
        if n.is_Integer:
            n = int(n)
            if n < 0:
                raise ValueError('Tribonacci polynomials are defined only for non-negative integer indices.')
            if sym is None:
                return Integer(cls._trib(n))
            else:
                return cls._tribpoly(n).subs(_sym, sym)

    def _eval_rewrite_as_sqrt(self, n, **kwargs):
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        w = (-1 + S.ImaginaryUnit * sqrt(3)) / 2
        a = (1 + cbrt(19 + 3 * sqrt(33)) + cbrt(19 - 3 * sqrt(33))) / 3
        b = (1 + w * cbrt(19 + 3 * sqrt(33)) + w ** 2 * cbrt(19 - 3 * sqrt(33))) / 3
        c = (1 + w ** 2 * cbrt(19 + 3 * sqrt(33)) + w * cbrt(19 - 3 * sqrt(33))) / 3
        Tn = a ** (n + 1) / ((a - b) * (a - c)) + b ** (n + 1) / ((b - a) * (b - c)) + c ** (n + 1) / ((c - a) * (c - b))
        return Tn

    def _eval_rewrite_as_TribonacciConstant(self, n, **kwargs):
        from sympy.functions.elementary.integers import floor
        from sympy.functions.elementary.miscellaneous import cbrt, sqrt
        b = cbrt(586 + 102 * sqrt(33))
        Tn = 3 * b * S.TribonacciConstant ** n / (b ** 2 - 2 * b + 4)
        return floor(Tn + S.Half)