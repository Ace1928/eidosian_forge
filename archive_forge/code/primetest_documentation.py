from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
Test if num is a Gaussian prime number.

    References
    ==========

    .. [1] https://oeis.org/wiki/Gaussian_primes
    