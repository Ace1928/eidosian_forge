import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class BoundedNumber(NumberGenerator):
    """
    Function object that silently enforces numeric bounds on values
    returned by a callable object.
    """
    generator = param.Callable(None, doc='Object to call to generate values.')
    bounds = param.Parameter((None, None), doc='\n        Legal range for the value returned, as a pair.\n\n        The default bounds are (None,None), meaning there are actually\n        no bounds.  One or both bounds can be set by specifying a\n        value.  For instance, bounds=(None,10) means there is no lower\n        bound, and an upper bound of 10.')

    def __call__(self):
        val = self.generator()
        min_, max_ = self.bounds
        if min_ is not None and val < min_:
            return min_
        elif max_ is not None and val > max_:
            return max_
        else:
            return val