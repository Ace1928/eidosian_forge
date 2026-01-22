import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_floor(self, prec):
    """Rounds down (not towards 0 if negative)"""
    if not self._sign:
        return self._round_down(prec)
    else:
        return -self._round_down(prec)