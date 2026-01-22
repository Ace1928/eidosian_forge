import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class OnlyOnce(object):
    """Wrapper for parse actions, to ensure they are only called once."""

    def __init__(self, methodCall):
        self.callable = _trim_arity(methodCall)
        self.called = False

    def __call__(self, s, l, t):
        if not self.called:
            results = self.callable(s, l, t)
            self.called = True
            return results
        raise ParseException(s, l, '')

    def reset(self):
        self.called = False