import contextlib
import copy
import enum
import functools
import inspect
import itertools
import linecache
import sys
import types
import typing
from operator import itemgetter
from . import _compat, _config, setters
from ._compat import (
from .exceptions import (
def _determine_whether_to_implement(cls, flag, auto_detect, dunders, default=True):
    """
    Check whether we should implement a set of methods for *cls*.

    *flag* is the argument passed into @attr.s like 'init', *auto_detect* the
    same as passed into @attr.s and *dunders* is a tuple of attribute names
    whose presence signal that the user has implemented it themselves.

    Return *default* if no reason for either for or against is found.
    """
    if flag is True or flag is False:
        return flag
    if flag is None and auto_detect is False:
        return default
    for dunder in dunders:
        if _has_own_attribute(cls, dunder):
            return False
    return default