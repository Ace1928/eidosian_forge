import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _set_integer_check(self, name, value, vmin, vmax):
    if not isinstance(value, int):
        raise TypeError('%s must be an integer' % name)
    if vmin == '-inf':
        if value > vmax:
            raise ValueError('%s must be in [%s, %d]. got: %s' % (name, vmin, vmax, value))
    elif vmax == 'inf':
        if value < vmin:
            raise ValueError('%s must be in [%d, %s]. got: %s' % (name, vmin, vmax, value))
    elif value < vmin or value > vmax:
        raise ValueError('%s must be in [%d, %d]. got %s' % (name, vmin, vmax, value))
    return object.__setattr__(self, name, value)