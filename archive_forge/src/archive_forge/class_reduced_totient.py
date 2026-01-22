from collections import defaultdict
from functools import reduce
import random
import math
from sympy.core import sympify
from sympy.core.containers import Dict
from sympy.core.evalf import bitcount
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.numbers import igcd, ilcm, Rational, Integer
from sympy.core.power import integer_nthroot, Pow, integer_log
from sympy.core.singleton import S
from sympy.external.gmpy import SYMPY_INTS
from .primetest import isprime
from .generate import sieve, primerange, nextprime
from .digits import digits
from sympy.utilities.iterables import flatten
from sympy.utilities.misc import as_int, filldedent
from .ecm import _ecm_one_factor
class reduced_totient(Function):
    """
    Calculate the Carmichael reduced totient function lambda(n)

    ``reduced_totient(n)`` or `\\lambda(n)` is the smallest m > 0 such that
    `k^m \\equiv 1 \\mod n` for all k relatively prime to n.

    Examples
    ========

    >>> from sympy.ntheory import reduced_totient
    >>> reduced_totient(1)
    1
    >>> reduced_totient(8)
    2
    >>> reduced_totient(30)
    4

    See Also
    ========

    totient

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Carmichael_function
    .. [2] https://mathworld.wolfram.com/CarmichaelFunction.html

    """

    @classmethod
    def eval(cls, n):
        if n.is_Integer:
            if n < 1:
                raise ValueError('n must be a positive integer')
            factors = factorint(n)
            return cls._from_factors(factors)

    @classmethod
    def _from_factors(self, factors):
        """Subroutine to compute totient from already-computed factors
        """
        t = 1
        for p, k in factors.items():
            if p == 2 and k > 2:
                t = ilcm(t, 2 ** (k - 2))
            else:
                t = ilcm(t, (p - 1) * p ** (k - 1))
        return t

    @classmethod
    def _from_distinct_primes(self, *args):
        """Subroutine to compute totient from the list of assumed
        distinct primes
        """
        args = [p - 1 for p in args]
        return ilcm(*args)

    def _eval_is_integer(self):
        return fuzzy_and([self.args[0].is_integer, self.args[0].is_positive])