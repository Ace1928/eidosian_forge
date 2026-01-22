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
class andre(Function):
    """
    Andre numbers / Andre function

    The Andre number `\\mathcal{A}_n` is Luschny's name for half the number of
    *alternating permutations* on `n` elements, where a permutation is alternating
    if adjacent elements alternately compare "greater" and "smaller" going from
    left to right. For example, `2 < 3 > 1 < 4` is an alternating permutation.

    This sequence is A000111 in the OEIS, which assigns the names *up/down numbers*
    and *Euler zigzag numbers*. It satisfies a recurrence relation similar to that
    for the Catalan numbers, with `\\mathcal{A}_0 = 1` and

    .. math:: 2 \\mathcal{A}_{n+1} = \\sum_{k=0}^n \\binom{n}{k} \\mathcal{A}_k \\mathcal{A}_{n-k}

    The Bernoulli and Euler numbers are signed transformations of the odd- and
    even-indexed elements of this sequence respectively:

    .. math :: \\operatorname{B}_{2k} = \\frac{2k \\mathcal{A}_{2k-1}}{(-4)^k - (-16)^k}

    .. math :: \\operatorname{E}_{2k} = (-1)^k \\mathcal{A}_{2k}

    Like the Bernoulli and Euler numbers, the Andre numbers are interpolated by the
    entire Andre function:

    .. math :: \\mathcal{A}(s) = (-i)^{s+1} \\operatorname{Li}_{-s}(i) +
            i^{s+1} \\operatorname{Li}_{-s}(-i) = \\\\ \\frac{2 \\Gamma(s+1)}{(2\\pi)^{s+1}}
            (\\zeta(s+1, 1/4) - \\zeta(s+1, 3/4) \\cos{\\pi s})

    Examples
    ========

    >>> from sympy import andre, euler, bernoulli
    >>> [andre(n) for n in range(11)]
    [1, 1, 1, 2, 5, 16, 61, 272, 1385, 7936, 50521]
    >>> [(-1)**k * andre(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [euler(2*k) for k in range(7)]
    [1, -1, 5, -61, 1385, -50521, 2702765]
    >>> [andre(2*k-1) * (2*k) / ((-4)**k - (-16)**k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]
    >>> [bernoulli(2*k) for k in range(1, 8)]
    [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6]

    See Also
    ========

    bernoulli, catalan, euler, sympy.polys.appellseqs.andre_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [2] https://mathworld.wolfram.com/EulerZigzagNumber.html
    .. [3] Peter Luschny, "An introduction to the Bernoulli function",
           https://arxiv.org/abs/2009.06743
    """

    @classmethod
    def eval(cls, n):
        if n is S.NaN:
            return S.NaN
        elif n is S.Infinity:
            return S.Infinity
        if n.is_zero:
            return S.One
        elif n == -1:
            return -log(2)
        elif n == -2:
            return -2 * S.Catalan
        elif n.is_Integer:
            if n.is_nonnegative and n.is_even:
                return abs(euler(n))
            elif n.is_odd:
                from sympy.functions.special.zeta_functions import zeta
                m = -n - 1
                return I ** m * Rational(1 - 2 ** m, 4 ** m) * zeta(-n)

    def _eval_rewrite_as_zeta(self, s, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        from sympy.functions.special.gamma_functions import gamma
        from sympy.functions.special.zeta_functions import zeta
        return 2 * gamma(s + 1) / (2 * pi) ** (s + 1) * (zeta(s + 1, S.One / 4) - cos(pi * s) * zeta(s + 1, S(3) / 4))

    def _eval_rewrite_as_polylog(self, s, **kwargs):
        from sympy.functions.special.zeta_functions import polylog
        return (-I) ** (s + 1) * polylog(-s, I) + I ** (s + 1) * polylog(-s, -I)

    def _eval_is_integer(self):
        n = self.args[0]
        if n.is_integer and n.is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_evalf(self, prec):
        if not self.args[0].is_number:
            return
        s = self.args[0]._to_mpmath(prec + 12)
        with workprec(prec + 12):
            sp, cp = (mp.sinpi(s / 2), mp.cospi(s / 2))
            res = 2 * mp.dirichlet(-s, (-sp, cp, sp, -cp))
        return Expr._from_mpmath(res, prec)