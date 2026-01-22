import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
def _deprecate(original_func):

    def deprecated_func(*args, **kwargs):
        warnings.warn('Please use {0} instead.'.format(original_func.__name__), DeprecationWarning, 2)
        return original_func(*args, **kwargs)
    return deprecated_func