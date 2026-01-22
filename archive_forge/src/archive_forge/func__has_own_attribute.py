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
def _has_own_attribute(cls, attrib_name):
    """
    Check whether *cls* defines *attrib_name* (and doesn't just inherit it).
    """
    attr = getattr(cls, attrib_name, _sentinel)
    if attr is _sentinel:
        return False
    for base_cls in cls.__mro__[1:]:
        a = getattr(base_cls, attrib_name, None)
        if attr is a:
            return False
    return True