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
class genocchi(Function):
    """
    Genocchi numbers / Genocchi polynomials / Genocchi function

    The Genocchi numbers are a sequence of integers `G_n` that satisfy the
    relation:

    .. math:: \\frac{-2t}{1 + e^{-t}} = \\sum_{n=0}^\\infty \\frac{G_n t^n}{n!}

    They are related to the Bernoulli numbers by

    .. math:: G_n = 2 (1 - 2^n) B_n

    and generalize like the Bernoulli numbers to the Genocchi polynomials and
    function as

    .. math:: \\operatorname{G}(s, a) = 2 \\left(\\operatorname{B}(s, a) -
              2^s \\operatorname{B}\\left(s, \\frac{a+1}{2}\\right)\\right)

    .. versionchanged:: 1.12
        ``genocchi(1)`` gives `-1` instead of `1`.

    Examples
    ========

    >>> from sympy import genocchi, Symbol
    >>> [genocchi(n) for n in range(9)]
    [0, -1, -1, 0, 1, 0, -3, 0, 17]
    >>> n = Symbol('n', integer=True, positive=True)
    >>> genocchi(2*n + 1)
    0
    >>> x = Symbol('x')
    >>> genocchi(4, x)
    -4*x**3 + 6*x**2 - 1

    See Also
    ========

    bell, bernoulli, catalan, euler, fibonacci, harmonic, lucas, partition, tribonacci
    sympy.polys.appellseqs.genocchi_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Genocchi_number
    .. [2] https://mathworld.wolfram.com/GenocchiNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743

    """

    @classmethod
    def eval(cls, n, x=None):
        if x is S.One:
            return cls(n)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        elif x is None:
            if n.is_odd and (n - 1).is_positive:
                return S.Zero
            elif n.is_Number:
                return 2 * (1 - S(2) ** n) * bernoulli(n)
        elif n.is_Number:
            return genocchi_poly(n, x)

    def _eval_rewrite_as_bernoulli(self, n, x=1, **kwargs):
        if x == 1 and n.is_integer and n.is_nonnegative:
            return 2 * (1 - S(2) ** n) * bernoulli(n)
        return 2 * (bernoulli(n, x) - 2 ** n * bernoulli(n, (x + 1) / 2))

    def _eval_rewrite_as_dirichlet_eta(self, n, x=1, **kwargs):
        from sympy.functions.special.zeta_functions import dirichlet_eta
        return -2 * n * dirichlet_eta(1 - n, x)

    def _eval_is_integer(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_negative(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_odd:
                return fuzzy_not((n - 1).is_positive)
            return (n / 2).is_odd

    def _eval_is_positive(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_zero or n.is_odd:
                return False
            return (n / 2).is_even

    def _eval_is_even(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return n.is_zero
            return (n - 1).is_positive

    def _eval_is_odd(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            if n.is_even:
                return fuzzy_not(n.is_zero)
            return fuzzy_not((n - 1).is_positive)

    def _eval_is_prime(self):
        if len(self.args) > 1 and self.args[1] != 1:
            return
        n = self.args[0]
        return (n - 8).is_zero

    def _eval_evalf(self, prec):
        if all((i.is_number for i in self.args)):
            return self.rewrite(bernoulli)._eval_evalf(prec)