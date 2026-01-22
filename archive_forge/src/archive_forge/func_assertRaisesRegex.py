from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def assertRaisesRegex(self, *args, **kwargs):
    return getattr(self, _assertRaisesRegex)(*args, **kwargs)