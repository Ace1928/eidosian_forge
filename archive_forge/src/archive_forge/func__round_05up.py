import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _round_05up(self, prec):
    """Round down unless digit prec-1 is 0 or 5."""
    if prec and self._int[prec - 1] not in '05':
        return self._round_down(prec)
    else:
        return -self._round_down(prec)