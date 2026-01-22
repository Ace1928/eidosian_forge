import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_half_even(self, prec):
    """Round 5 to even, rest to nearest."""
    if _exact_half(self._int, prec) and (prec == 0 or self._int[prec - 1] in '02468'):
        return -1
    else:
        return self._round_half_up(prec)