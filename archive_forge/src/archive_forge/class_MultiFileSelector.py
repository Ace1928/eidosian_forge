import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
class MultiFileSelector(ListSelector):
    """
    Given a path glob, allows multiple files to be selected from the list of matches.
    """
    __slots__ = ['path']
    _slot_defaults = _dict_update(Selector._slot_defaults, path='')

    @typing.overload
    def __init__(self, default=None, *, path='', objects=[], compute_default_fn=None, check_on_set=None, allow_None=None, empty_default=False, doc=None, label=None, precedence=None, instantiate=False, constant=False, readonly=False, pickle_default_value=True, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    @_deprecate_positional_args
    def __init__(self, default=Undefined, *, path=Undefined, **kwargs):
        self.default = default
        self.path = path
        self.update(path=path)
        super().__init__(default=default, objects=self._objects, **kwargs)

    def _on_set(self, attribute, old, new):
        super()._on_set(attribute, new, old)
        if attribute == 'path':
            self.update(path=new)

    def update(self, path=Undefined):
        if path is Undefined:
            path = self.path
        self.objects = sorted(glob.glob(path))
        if self.default and all([o in self.objects for o in self.default]):
            return
        elif not self.default:
            return
        self.default = self.objects

    def get_range(self):
        return _abbreviate_paths(self.path, super().get_range())