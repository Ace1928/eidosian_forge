import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _set_signal_dict(self, name, d):
    if not isinstance(d, dict):
        raise TypeError('%s must be a signal dict' % d)
    for key in d:
        if not key in _signals:
            raise KeyError('%s is not a valid signal dict' % d)
    for key in _signals:
        if not key in d:
            raise KeyError('%s is not a valid signal dict' % d)
    return object.__setattr__(self, name, d)