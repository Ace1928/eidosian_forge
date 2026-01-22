import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _iseven(self):
    """Returns True if self is even.  Assumes self is an integer."""
    if not self or self._exp > 0:
        return True
    return self._int[-1 + self._exp] in '02468'