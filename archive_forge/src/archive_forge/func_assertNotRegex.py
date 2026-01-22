from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def assertNotRegex(self, *args, **kwargs):
    return getattr(self, _assertNotRegex)(*args, **kwargs)