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
class bell(Function):
    """
    Bell numbers / Bell polynomials

    The Bell numbers satisfy `B_0 = 1` and

    .. math:: B_n = \\sum_{k=0}^{n-1} \\binom{n-1}{k} B_k.

    They are also given by:

    .. math:: B_n = \\frac{1}{e} \\sum_{k=0}^{\\infty} \\frac{k^n}{k!}.

    The Bell polynomials are given by `B_0(x) = 1` and

    .. math:: B_n(x) = x \\sum_{k=1}^{n-1} \\binom{n-1}{k-1} B_{k-1}(x).

    The second kind of Bell polynomials (are sometimes called "partial" Bell
    polynomials or incomplete Bell polynomials) are defined as

    .. math:: B_{n,k}(x_1, x_2,\\dotsc x_{n-k+1}) =
            \\sum_{j_1+j_2+j_2+\\dotsb=k \\atop j_1+2j_2+3j_2+\\dotsb=n}
                \\frac{n!}{j_1!j_2!\\dotsb j_{n-k+1}!}
                \\left(\\frac{x_1}{1!} \\right)^{j_1}
                \\left(\\frac{x_2}{2!} \\right)^{j_2} \\dotsb
                \\left(\\frac{x_{n-k+1}}{(n-k+1)!} \\right) ^{j_{n-k+1}}.

    * ``bell(n)`` gives the `n^{th}` Bell number, `B_n`.
    * ``bell(n, x)`` gives the `n^{th}` Bell polynomial, `B_n(x)`.
    * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
      `B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})`.

    Notes
    =====

    Not to be confused with Bernoulli numbers and Bernoulli polynomials,
    which use the same notation.

    Examples
    ========

    >>> from sympy import bell, Symbol, symbols

    >>> [bell(n) for n in range(11)]
    [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    >>> bell(30)
    846749014511809332450147
    >>> bell(4, Symbol('t'))
    t**4 + 6*t**3 + 7*t**2 + t
    >>> bell(6, 2, symbols('x:6')[1:])
    6*x1*x5 + 15*x2*x4 + 10*x3**2

    See Also
    ========

    bernoulli, catalan, euler, fibonacci, harmonic, lucas, genocchi, partition, tribonacci

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bell_number
    .. [2] https://mathworld.wolfram.com/BellNumber.html
    .. [3] https://mathworld.wolfram.com/BellPolynomial.html

    """

    @staticmethod
    @recurrence_memo([1, 1])
    def _bell(n, prev):
        s = 1
        a = 1
        for k in range(1, n):
            a = a * (n - k) // k
            s += a * prev[k]
        return s

    @staticmethod
    @recurrence_memo([S.One, _sym])
    def _bell_poly(n, prev):
        s = 1
        a = 1
        for k in range(2, n + 1):
            a = a * (n - k + 1) // (k - 1)
            s += a * prev[k - 1]
        return expand_mul(_sym * s)

    @staticmethod
    def _bell_incomplete_poly(n, k, symbols):
        """
        The second kind of Bell polynomials (incomplete Bell polynomials).

        Calculated by recurrence formula:

        .. math:: B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1}) =
                \\sum_{m=1}^{n-k+1}
                \\x_m \\binom{n-1}{m-1} B_{n-m,k-1}(x_1, x_2, \\dotsc, x_{n-m-k})

        where
            `B_{0,0} = 1;`
            `B_{n,0} = 0; for n \\ge 1`
            `B_{0,k} = 0; for k \\ge 1`

        """
        if n == 0 and k == 0:
            return S.One
        elif n == 0 or k == 0:
            return S.Zero
        s = S.Zero
        a = S.One
        for m in range(1, n - k + 2):
            s += a * bell._bell_incomplete_poly(n - m, k - 1, symbols) * symbols[m - 1]
            a = a * (n - m) / m
        return expand_mul(s)

    @classmethod
    def eval(cls, n, k_sym=None, symbols=None):
        if n is S.Infinity:
            if k_sym is None:
                return S.Infinity
            else:
                raise ValueError('Bell polynomial is not defined')
        if n.is_negative or n.is_integer is False:
            raise ValueError('a non-negative integer expected')
        if n.is_Integer and n.is_nonnegative:
            if k_sym is None:
                return Integer(cls._bell(int(n)))
            elif symbols is None:
                return cls._bell_poly(int(n)).subs(_sym, k_sym)
            else:
                r = cls._bell_incomplete_poly(int(n), int(k_sym), symbols)
                return r

    def _eval_rewrite_as_Sum(self, n, k_sym=None, symbols=None, **kwargs):
        from sympy.concrete.summations import Sum
        if k_sym is not None or symbols is not None:
            return self
        if not n.is_nonnegative:
            return self
        k = Dummy('k', integer=True, nonnegative=True)
        return 1 / E * Sum(k ** n / factorial(k), (k, 0, S.Infinity))