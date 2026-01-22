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
class catalan(Function):
    """
    Catalan numbers

    The `n^{th}` catalan number is given by:

    .. math :: C_n = \\frac{1}{n+1} \\binom{2n}{n}

    * ``catalan(n)`` gives the `n^{th}` Catalan number, `C_n`

    Examples
    ========

    >>> from sympy import (Symbol, binomial, gamma, hyper,
    ...     catalan, diff, combsimp, Rational, I)

    >>> [catalan(i) for i in range(1,10)]
    [1, 2, 5, 14, 42, 132, 429, 1430, 4862]

    >>> n = Symbol("n", integer=True)

    >>> catalan(n)
    catalan(n)

    Catalan numbers can be transformed into several other, identical
    expressions involving other mathematical functions

    >>> catalan(n).rewrite(binomial)
    binomial(2*n, n)/(n + 1)

    >>> catalan(n).rewrite(gamma)
    4**n*gamma(n + 1/2)/(sqrt(pi)*gamma(n + 2))

    >>> catalan(n).rewrite(hyper)
    hyper((1 - n, -n), (2,), 1)

    For some non-integer values of n we can get closed form
    expressions by rewriting in terms of gamma functions:

    >>> catalan(Rational(1, 2)).rewrite(gamma)
    8/(3*pi)

    We can differentiate the Catalan numbers C(n) interpreted as a
    continuous real function in n:

    >>> diff(catalan(n), n)
    (polygamma(0, n + 1/2) - polygamma(0, n + 2) + log(4))*catalan(n)

    As a more advanced example consider the following ratio
    between consecutive numbers:

    >>> combsimp((catalan(n + 1)/catalan(n)).rewrite(binomial))
    2*(2*n + 1)/(n + 2)

    The Catalan numbers can be generalized to complex numbers:

    >>> catalan(I).rewrite(gamma)
    4**I*gamma(1/2 + I)/(sqrt(pi)*gamma(2 + I))

    and evaluated with arbitrary precision:

    >>> catalan(I).evalf(20)
    0.39764993382373624267 - 0.020884341620842555705*I

    See Also
    ========

    andre, bell, bernoulli, euler, fibonacci, harmonic, lucas, genocchi,
    partition, tribonacci, sympy.functions.combinatorial.factorials.binomial

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Catalan_number
    .. [2] https://mathworld.wolfram.com/CatalanNumber.html
    .. [3] https://functions.wolfram.com/GammaBetaErf/CatalanNumber/
    .. [4] http://geometer.org/mathcircles/catalan.pdf

    """

    @classmethod
    def eval(cls, n):
        from sympy.functions.special.gamma_functions import gamma
        if n.is_Integer and n.is_nonnegative or (n.is_noninteger and n.is_negative):
            return 4 ** n * gamma(n + S.Half) / (gamma(S.Half) * gamma(n + 2))
        if n.is_integer and n.is_negative:
            if (n + 1).is_negative:
                return S.Zero
            if (n + 1).is_zero:
                return Rational(-1, 2)

    def fdiff(self, argindex=1):
        from sympy.functions.elementary.exponential import log
        from sympy.functions.special.gamma_functions import polygamma
        n = self.args[0]
        return catalan(n) * (polygamma(0, n + S.Half) - polygamma(0, n + 2) + log(4))

    def _eval_rewrite_as_binomial(self, n, **kwargs):
        return binomial(2 * n, n) / (n + 1)

    def _eval_rewrite_as_factorial(self, n, **kwargs):
        return factorial(2 * n) / (factorial(n + 1) * factorial(n))

    def _eval_rewrite_as_gamma(self, n, piecewise=True, **kwargs):
        from sympy.functions.special.gamma_functions import gamma
        return 4 ** n * gamma(n + S.Half) / (gamma(S.Half) * gamma(n + 2))

    def _eval_rewrite_as_hyper(self, n, **kwargs):
        from sympy.functions.special.hyper import hyper
        return hyper([1 - n, -n], [2], 1)

    def _eval_rewrite_as_Product(self, n, **kwargs):
        from sympy.concrete.products import Product
        if not (n.is_integer and n.is_nonnegative):
            return self
        k = Dummy('k', integer=True, positive=True)
        return Product((n + k) / k, (k, 2, n))

    def _eval_is_integer(self):
        if self.args[0].is_integer and self.args[0].is_nonnegative:
            return True

    def _eval_is_positive(self):
        if self.args[0].is_nonnegative:
            return True

    def _eval_is_composite(self):
        if self.args[0].is_integer and (self.args[0] - 3).is_positive:
            return True

    def _eval_evalf(self, prec):
        from sympy.functions.special.gamma_functions import gamma
        if self.args[0].is_number:
            return self.rewrite(gamma)._eval_evalf(prec)