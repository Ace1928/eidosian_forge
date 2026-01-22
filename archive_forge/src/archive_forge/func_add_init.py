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
def add_init(self):
    self._cls_dict['__init__'] = self._add_method_dunders(_make_init(self._cls, self._attrs, self._has_pre_init, self._pre_init_has_args, self._has_post_init, self._frozen, self._slots, self._cache_hash, self._base_attr_map, self._is_exc, self._on_setattr, attrs_init=False))
    return self