from sympy.core.numbers import igcd, mod_inverse
from sympy.core.power import integer_nthroot
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power
from sympy.ntheory import isprime
from math import log, sqrt
import random
class SievePolynomial:

    def __init__(self, modified_coeff=(), a=None, b=None):
        """This class denotes the seive polynomial.
        If ``g(x) = (a*x + b)**2 - N``. `g(x)` can be expanded
        to ``a*x**2 + 2*a*b*x + b**2 - N``, so the coefficient
        is stored in the form `[a**2, 2*a*b, b**2 - N]`. This
        ensures faster `eval` method because we dont have to
        perform `a**2, 2*a*b, b**2` every time we call the
        `eval` method. As multiplication is more expensive
        than addition, by using modified_coefficient we get
        a faster seiving process.

        Parameters
        ==========

        modified_coeff : modified_coefficient of sieve polynomial
        a : parameter of the sieve polynomial
        b : parameter of the sieve polynomial
        """
        self.modified_coeff = modified_coeff
        self.a = a
        self.b = b

    def eval(self, x):
        """
        Compute the value of the sieve polynomial at point x.

        Parameters
        ==========

        x : Integer parameter for sieve polynomial
        """
        ans = 0
        for coeff in self.modified_coeff:
            ans *= x
            ans += coeff
        return ans