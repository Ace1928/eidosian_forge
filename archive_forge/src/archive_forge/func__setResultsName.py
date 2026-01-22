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
def _setResultsName(self, name, listAllMatches=False):
    if __diag__.warn_name_set_on_empty_Forward:
        if self.expr is None:
            warnings.warn('{0}: setting results name {0!r} on {1} expression that has no contained expression'.format('warn_name_set_on_empty_Forward', name, type(self).__name__), stacklevel=3)
    return super(Forward, self)._setResultsName(name, listAllMatches)