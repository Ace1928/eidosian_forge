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
def _baseAssertEqual(self, first, second, msg=None):
    """The default assertEqual implementation, not type specific."""
    if not first == second:
        standardMsg = '%s != %s' % _common_shorten_repr(first, second)
        msg = self._formatMessage(msg, standardMsg)
        raise self.failureException(msg)