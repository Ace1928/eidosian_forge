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
def _truncateMessage(self, message, diff):
    max_diff = self.maxDiff
    if max_diff is None or len(diff) <= max_diff:
        return message + diff
    return message + DIFF_OMITTED % len(diff)