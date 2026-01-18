import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def to_eng_string(self, a):
    """Convert to a string, using engineering notation if an exponent is needed.

        Engineering notation has an exponent which is a multiple of 3.  This
        can leave up to 3 digits to the left of the decimal place and may
        require the addition of either one or two trailing zeros.

        The operation is not affected by the context.

        >>> ExtendedContext.to_eng_string(Decimal('123E+1'))
        '1.23E+3'
        >>> ExtendedContext.to_eng_string(Decimal('123E+3'))
        '123E+3'
        >>> ExtendedContext.to_eng_string(Decimal('123E-10'))
        '12.3E-9'
        >>> ExtendedContext.to_eng_string(Decimal('-123E-12'))
        '-123E-12'
        >>> ExtendedContext.to_eng_string(Decimal('7E-7'))
        '700E-9'
        >>> ExtendedContext.to_eng_string(Decimal('7E+1'))
        '70'
        >>> ExtendedContext.to_eng_string(Decimal('0E+1'))
        '0.00E+3'

        """
    a = _convert_other(a, raiseit=True)
    return a.to_eng_string(context=self)