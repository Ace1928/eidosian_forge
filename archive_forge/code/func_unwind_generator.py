from . import version
import collections
from functools import wraps
import sys
import warnings
@wraps(func)
def unwind_generator(*args, **kwargs):
    return _inline_callbacks(None, func(*args, **kwargs), Deferred())