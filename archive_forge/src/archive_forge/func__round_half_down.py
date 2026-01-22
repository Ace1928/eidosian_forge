import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_half_down(self, prec):
    """Round 5 down"""
    if _exact_half(self._int, prec):
        return -1
    else:
        return self._round_half_up(prec)