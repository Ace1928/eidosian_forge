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
def _patch_original_class(self):
    """
        Apply accumulated methods and return the class.
        """
    cls = self._cls
    base_names = self._base_names
    if self._delete_attribs:
        for name in self._attr_names:
            if name not in base_names and getattr(cls, name, _sentinel) is not _sentinel:
                with contextlib.suppress(AttributeError):
                    delattr(cls, name)
    for name, value in self._cls_dict.items():
        setattr(cls, name, value)
    if not self._wrote_own_setattr and getattr(cls, '__attrs_own_setattr__', False):
        cls.__attrs_own_setattr__ = False
        if not self._has_custom_setattr:
            cls.__setattr__ = _obj_setattr
    return cls