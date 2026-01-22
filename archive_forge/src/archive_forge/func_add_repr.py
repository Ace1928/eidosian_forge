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
def add_repr(self, ns):
    self._cls_dict['__repr__'] = self._add_method_dunders(_make_repr(self._attrs, ns, self._cls))
    return self