from __future__ import absolute_import
import functools
import itertools
import operator
import sys
import types
def _add_doc(func, doc):
    """Add documentation to a function."""
    func.__doc__ = doc