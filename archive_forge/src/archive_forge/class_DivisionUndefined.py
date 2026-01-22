import math as _math
import numbers as _numbers
import sys
import contextvars
import re
class DivisionUndefined(InvalidOperation, ZeroDivisionError):
    """Undefined result of division.

    This occurs and signals invalid-operation if division by zero was
    attempted (during a divide-integer, divide, or remainder operation), and
    the dividend is also zero.  The result is [0,qNaN].
    """

    def handle(self, context, *args):
        return _NaN