from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def to_PUR(self):
    """
        Converts it into a Polynomial Univariate Representation that is
        either a pari POLMOD object or a pari rational number.

        >>> nf = pari("x^9+x^3+x+32121")
        >>> a = pari("43*x^3 + 1")
        >>> b = pari("x^2 + 3")

        >>> r = RUR.from_pari_fraction_and_number_field(a / b, nf)
        >>> r
        ( Mod(43*x^3 + 1, x^9 + x^3 + x + 32121) ) / ( Mod(x^2 + 3, x^9 + x^3 + x + 32121) )

        Polynomial Univariate Representation has larger coefficients:

        >>> r.to_PUR()
        Mod(1035922/257944341*x^8 - 129/85981447*x^7 - 1035922/85981447*x^6 + 387/85981447*x^5 + 3107766/85981447*x^4 - 1161/85981447*x^3 - 26933972/257944341*x^2 + 3697205575/85981447*x + 81837838/257944341, x^9 + x^3 + x + 32121)


        """
    return pari(self._numerator()) / pari(self._denominator())