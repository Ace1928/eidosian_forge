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
def assertIn(self, member, container, msg=None):
    """Just like self.assertTrue(a in b), but with a nicer default message."""
    if member not in container:
        standardMsg = '%s not found in %s' % (safe_repr(member), safe_repr(container))
        self.fail(self._formatMessage(msg, standardMsg))