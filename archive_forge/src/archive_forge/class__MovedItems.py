from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
class _MovedItems(_LazyModule):
    """Lazy loading of moved objects"""
    __path__ = []