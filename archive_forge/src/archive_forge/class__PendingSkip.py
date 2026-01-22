import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
import pprint
import traceback
import types
from datetime import datetime
from operator import itemgetter
import itertools
from functools import wraps
from contextlib import contextmanager
class _PendingSkip(ParserElement):

    def __init__(self, expr, must_skip=False):
        super(_PendingSkip, self).__init__()
        self.strRepr = str(expr + Empty()).replace('Empty', '...')
        self.name = self.strRepr
        self.anchor = expr
        self.must_skip = must_skip

    def __add__(self, other):
        skipper = SkipTo(other).setName('...')('_skipped*')
        if self.must_skip:

            def must_skip(t):
                if not t._skipped or t._skipped.asList() == ['']:
                    del t[0]
                    t.pop('_skipped', None)

            def show_skip(t):
                if t._skipped.asList()[-1:] == ['']:
                    skipped = t.pop('_skipped')
                    t['_skipped'] = 'missing <' + repr(self.anchor) + '>'
            return (self.anchor + skipper().addParseAction(must_skip) | skipper().addParseAction(show_skip)) + other
        return self.anchor + skipper + other

    def __repr__(self):
        return self.strRepr

    def parseImpl(self, *args):
        raise Exception('use of `...` expression without following SkipTo target expression')