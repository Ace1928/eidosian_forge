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
def assertSetEqual(self, set1, set2, msg=None):
    """A set-specific equality assertion.

        Args:
            set1: The first set to compare.
            set2: The second set to compare.
            msg: Optional message to use on failure instead of a list of
                    differences.

        assertSetEqual uses ducktyping to support different types of sets, and
        is optimized for sets specifically (parameters must support a
        difference method).
        """
    try:
        difference1 = set1.difference(set2)
    except TypeError as e:
        self.fail('invalid type when attempting set difference: %s' % e)
    except AttributeError as e:
        self.fail('first argument does not support set difference: %s' % e)
    try:
        difference2 = set2.difference(set1)
    except TypeError as e:
        self.fail('invalid type when attempting set difference: %s' % e)
    except AttributeError as e:
        self.fail('second argument does not support set difference: %s' % e)
    if not (difference1 or difference2):
        return
    lines = []
    if difference1:
        lines.append('Items in the first set but not the second:')
        for item in difference1:
            lines.append(repr(item))
    if difference2:
        lines.append('Items in the second set but not the first:')
        for item in difference2:
            lines.append(repr(item))
    standardMsg = '\n'.join(lines)
    self.fail(self._formatMessage(msg, standardMsg))