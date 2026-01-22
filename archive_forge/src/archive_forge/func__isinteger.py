import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _isinteger(self):
    """Returns whether self is an integer"""
    if self._is_special:
        return False
    if self._exp >= 0:
        return True
    rest = self._int[self._exp:]
    return rest == '0' * len(rest)