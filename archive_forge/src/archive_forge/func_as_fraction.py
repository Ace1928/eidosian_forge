from .z3 import *
from .z3core import *
from .z3printer import *
from fractions import Fraction
from .z3 import _get_ctx
def as_fraction(self):
    """ Return a numeral (that is a rational) as a Python Fraction.
        >>> Numeral("1/5").as_fraction()
        Fraction(1, 5)
        """
    assert self.is_rational()
    return Fraction(self.numerator().as_long(), self.denominator().as_long())