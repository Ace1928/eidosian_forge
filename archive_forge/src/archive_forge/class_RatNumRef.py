from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class RatNumRef(ArithRef):
    """Rational values."""

    def numerator(self):
        """ Return the numerator of a Z3 rational numeral.

        >>> is_rational_value(RealVal("3/5"))
        True
        >>> n = RealVal("3/5")
        >>> n.numerator()
        3
        >>> is_rational_value(Q(3,5))
        True
        >>> Q(3,5).numerator()
        3
        """
        return IntNumRef(Z3_get_numerator(self.ctx_ref(), self.as_ast()), self.ctx)

    def denominator(self):
        """ Return the denominator of a Z3 rational numeral.

        >>> is_rational_value(Q(3,5))
        True
        >>> n = Q(3,5)
        >>> n.denominator()
        5
        """
        return IntNumRef(Z3_get_denominator(self.ctx_ref(), self.as_ast()), self.ctx)

    def numerator_as_long(self):
        """ Return the numerator as a Python long.

        >>> v = RealVal(10000000000)
        >>> v
        10000000000
        >>> v + 1
        10000000000 + 1
        >>> v.numerator_as_long() + 1 == 10000000001
        True
        """
        return self.numerator().as_long()

    def denominator_as_long(self):
        """ Return the denominator as a Python long.

        >>> v = RealVal("1/3")
        >>> v
        1/3
        >>> v.denominator_as_long()
        3
        """
        return self.denominator().as_long()

    def is_int(self):
        return False

    def is_real(self):
        return True

    def is_int_value(self):
        return self.denominator().is_int() and self.denominator_as_long() == 1

    def as_long(self):
        _z3_assert(self.is_int_value(), 'Expected integer fraction')
        return self.numerator_as_long()

    def as_decimal(self, prec):
        """ Return a Z3 rational value as a string in decimal notation using at most `prec` decimal places.

        >>> v = RealVal("1/5")
        >>> v.as_decimal(3)
        '0.2'
        >>> v = RealVal("1/3")
        >>> v.as_decimal(3)
        '0.333?'
        """
        return Z3_get_numeral_decimal_string(self.ctx_ref(), self.as_ast(), prec)

    def as_string(self):
        """Return a Z3 rational numeral as a Python string.

        >>> v = Q(3,6)
        >>> v.as_string()
        '1/2'
        """
        return Z3_get_numeral_string(self.ctx_ref(), self.as_ast())

    def as_fraction(self):
        """Return a Z3 rational as a Python Fraction object.

        >>> v = RealVal("1/5")
        >>> v.as_fraction()
        Fraction(1, 5)
        """
        return Fraction(self.numerator_as_long(), self.denominator_as_long())