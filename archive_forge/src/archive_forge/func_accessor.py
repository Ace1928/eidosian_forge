import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
@wraps(propfunc)
def accessor(self):
    val = getattr(self, attrname, sentinel)
    if val is sentinel:
        val = propfunc(self)
        setattr(self, attrname, val)
    return val