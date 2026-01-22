from sympy.core.numbers import igcd
from sympy.core.power import integer_nthroot
from sympy.core.sympify import sympify
from sympy.external.gmpy import HAS_GMPY
from sympy.utilities.misc import as_int
from mpmath.libmp import bitcount as _bitlength
def _lucas_selfridge_params(n):
    """Calculates the Selfridge parameters (D, P, Q) for n.  This is
       method A from page 1401 of Baillie and Wagstaff.

    References
    ==========
    .. [1] "Lucas Pseudoprimes", Baillie and Wagstaff, 1980.
           http://mpqs.free.fr/LucasPseudoprimes.pdf
    """
    from sympy.ntheory.residue_ntheory import jacobi_symbol
    D = 5
    while True:
        g = igcd(abs(D), n)
        if g > 1 and g != n:
            return (0, 0, 0)
        if jacobi_symbol(D, n) == -1:
            break
        if D > 0:
            D = -D - 2
        else:
            D = -D + 2
    return _int_tuple(D, 1, (1 - D) / 4)