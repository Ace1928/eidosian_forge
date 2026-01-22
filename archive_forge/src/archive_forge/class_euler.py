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
class euler(Function):
    """
    Euler numbers / Euler polynomials / Euler function

    The Euler numbers are given by:

    .. math:: E_{2n} = I \\sum_{k=1}^{2n+1} \\sum_{j=0}^k \\binom{k}{j}
        \\frac{(-1)^j (k-2j)^{2n+1}}{2^k I^k k}

    .. math:: E_{2n+1} = 0

    Euler numbers and Euler polynomials are related by

    .. math:: E_n = 2^n E_n\\left(\\frac{1}{2}\\right).

    We compute symbolic Euler polynomials using Appell sequences,
    but numerical evaluation of the Euler polynomial is computed
    more efficiently (and more accurately) using the mpmath library.

    The Euler polynomials are special cases of the generalized Euler function,
    related to the Genocchi function as

    .. math:: \\operatorname{E}(s, a) = -\\frac{\\operatorname{G}(s+1, a)}{s+1}

    with the limit of `\\psi\\left(\\frac{a+1}{2}\\right) - \\psi\\left(\\frac{a}{2}\\right)`
    being taken when `s = -1`. The (ordinary) Euler function interpolating
    the Euler numbers is then obtained as
    `\\operatorname{E}(s) = 2^s \\operatorname{E}\\left(s, \\frac{1}{2}\\right)`.

    * ``euler(n)`` gives the nth Euler number `E_n`.
    * ``euler(s)`` gives the Euler function `\\operatorname{E}(s)`.
    * ``euler(n, x)`` gives the nth Euler polynomial `E_n(x)`.
    * ``euler(s, a)`` gives the generalized Euler function `\\operatorname{E}(s, a)`.

    Examples
    ========

    >>> from sympy import euler, Symbol, S
    >>> [euler(n) for n in range(10)]
    [1, 0, -1, 0, 5, 0, -61, 0, 1385, 0]
    >>> [2**n*euler(n,1) for n in range(10)]
    [1, 1, 0, -2, 0, 16, 0, -272, 0, 7936]
    >>> n = Symbol("n")
    >>> euler(n + 2*n)
    euler(3*n)

    >>> x = Symbol("x")
    >>> euler(n, x)
    euler(n, x)

    >>> euler(0, x)
    1
    >>> euler(1, x)
    x - 1/2
    >>> euler(2, x)
    x**2 - x
    >>> euler(3, x)
    x**3 - 3*x**2/2 + 1/4
    >>> euler(4, x)
    x**4 - 2*x**3 + x

    >>> euler(12, S.Half)
    2702765/4096
    >>> euler(12)
    2702765

    See Also
    ========

    andre, bell, bernoulli, catalan, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.polys.appellseqs.euler_poly

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Euler_numbers
    .. [2] https://mathworld.wolfram.com/EulerNumber.html
    .. [3] https://en.wikipedia.org/wiki/Alternating_permutation
    .. [4] https://mathworld.wolfram.com/AlternatingPermutation.html

    """

    @classmethod
    def eval(cls, n, x=None):
        if n.is_zero:
            return S.One
        elif n is S.NegativeOne:
            if x is None:
                return S.Pi / 2
            from sympy.functions.special.gamma_functions import digamma
            return digamma((x + 1) / 2) - digamma(x / 2)
        elif n.is_integer is False or n.is_nonnegative is False:
            return
        elif x is None:
            if n.is_odd and n.is_positive:
                return S.Zero
            elif n.is_Number:
                from mpmath import mp
                n = n._to_mpmath(mp.prec)
                res = mp.eulernum(n, exact=True)
                return Integer(res)
        elif n.is_Number:
            return euler_poly(n, x)

    def _eval_rewrite_as_Sum(self, n, x=None, **kwargs):
        from sympy.concrete.summations import Sum
        if x is None and n.is_even:
            k = Dummy('k', integer=True)
            j = Dummy('j', integer=True)
            n = n / 2
            Em = S.ImaginaryUnit * Sum(Sum(binomial(k, j) * (S.NegativeOne ** j * (k - 2 * j) ** (2 * n + 1)) / (2 ** k * S.ImaginaryUnit ** k * k), (j, 0, k)), (k, 1, 2 * n + 1))
            return Em
        if x:
            k = Dummy('k', integer=True)
            return Sum(binomial(n, k) * euler(k) / 2 ** k * (x - S.Half) ** (n - k), (k, 0, n))

    def _eval_rewrite_as_genocchi(self, n, x=None, **kwargs):
        if x is None:
            return Piecewise((S.Pi / 2, Eq(n, -1)), (-2 ** n * genocchi(n + 1, S.Half) / (n + 1), True))
        from sympy.functions.special.gamma_functions import digamma
        return Piecewise((digamma((x + 1) / 2) - digamma(x / 2), Eq(n, -1)), (-genocchi(n + 1, x) / (n + 1), True))

    def _eval_evalf(self, prec):
        if not all((i.is_number for i in self.args)):
            return
        from mpmath import mp
        m, x = (self.args[0], None) if len(self.args) == 1 else self.args
        m = m._to_mpmath(prec)
        if x is not None:
            x = x._to_mpmath(prec)
        with workprec(prec):
            if mp.isint(m) and m >= 0:
                res = mp.eulernum(m) if x is None else mp.eulerpoly(m, x)
            else:
                if m == -1:
                    res = mp.pi if x is None else mp.digamma((x + 1) / 2) - mp.digamma(x / 2)
                else:
                    y = 0.5 if x is None else x
                    res = 2 * (mp.zeta(-m, y) - 2 ** (m + 1) * mp.zeta(-m, (y + 1) / 2))
                if x is None:
                    res *= 2 ** m
        return Expr._from_mpmath(res, prec)