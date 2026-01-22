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
def assertTrue(self, expr, msg=None):
    """Check that the expression is true."""
    if not expr:
        msg = self._formatMessage(msg, '%s is not true' % safe_repr(expr))
        raise self.failureException(msg)