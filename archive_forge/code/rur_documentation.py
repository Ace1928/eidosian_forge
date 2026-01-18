from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod

        Returns it as pari quotient of two polynomials. The value
        represented by this RUR is obtained by evaluating the quotient
        at a root of the polynomial returned by number_field.

        >>> nf = pari("x^97+x^3+x+32121")
        >>> a = pari("43*x^3 + 1")
        >>> b = pari("x^2 + 3")

        >>> r = 2 * RUR.from_pari_fraction_and_number_field(a / b, nf)
        >>> r
        ( Mod(43*x^3 + 1, x^97 + x^3 + x + 32121) * 2 ) / ( Mod(x^2 + 3, x^97 + x^3 + x + 32121) )

        >>> r.multiply_terms()
        ( Mod(86*x^3 + 2, x^97 + x^3 + x + 32121) ) / ( Mod(x^2 + 3, x^97 + x^3 + x + 32121) )

        >>> r.to_pari_fraction()
        (86*x^3 + 2)/(x^2 + 3)

        