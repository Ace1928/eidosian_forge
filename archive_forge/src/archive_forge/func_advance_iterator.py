from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def advance_iterator(it):
    return it.next()