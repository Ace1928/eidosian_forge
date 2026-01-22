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
def _callTestMethod(self, method):
    if method() is not None:
        warnings.warn(f'It is deprecated to return a value that is not None from a test case ({method})', DeprecationWarning, stacklevel=3)