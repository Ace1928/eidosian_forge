from __future__ import division
from future.utils import PYPY, PY26, bind_method
from decimal import Decimal, ROUND_HALF_EVEN
Converts a float to a decimal number, exactly.

    Note that Decimal.from_float(0.1) is not the same as Decimal('0.1').
    Since 0.1 is not exactly representable in binary floating point, the
    value is stored as the nearest representable value which is
    0x1.999999999999ap-4.  The exact equivalent of the value in decimal
    is 0.1000000000000000055511151231257827021181583404541015625.

    >>> Decimal.from_float(0.1)
    Decimal('0.1000000000000000055511151231257827021181583404541015625')
    >>> Decimal.from_float(float('nan'))
    Decimal('NaN')
    >>> Decimal.from_float(float('inf'))
    Decimal('Infinity')
    >>> Decimal.from_float(-float('inf'))
    Decimal('-Infinity')
    >>> Decimal.from_float(-0.0)
    Decimal('-0')

    