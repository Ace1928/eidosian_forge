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
class ListSelector(Selector):
    """
    Variant of Selector where the value can be multiple objects from
    a list of possible objects.
    """

    @typing.overload
    def __init__(self, default=None, *, objects=[], instantiate=False, compute_default_fn=None, check_on_set=None, allow_None=None, empty_default=False, doc=None, label=None, precedence=None, constant=False, readonly=False, pickle_default_value=True, per_instance=True, allow_refs=False, nested_refs=False):
        ...

    @_deprecate_positional_args
    def __init__(self, default=Undefined, *, objects=Undefined, **kwargs):
        super().__init__(objects=objects, default=default, empty_default=True, **kwargs)

    def compute_default(self):
        if self.default is None and callable(self.compute_default_fn):
            self.default = self.compute_default_fn()
            for o in self.default:
                if o not in self.objects:
                    self.objects.append(o)

    def _validate(self, val):
        if val is None and self.allow_None:
            return
        self._validate_type(val)
        if self.check_on_set:
            self._validate_value(val)
        else:
            for v in val:
                self._ensure_value_is_in_objects(v)

    def _validate_type(self, val):
        if not isinstance(val, list):
            raise ValueError(f'{_validate_error_prefix(self)} only takes list types, not {val!r}.')

    def _validate_value(self, val):
        self._validate_type(val)
        if val is not None:
            for o in val:
                super()._validate_value(o)

    def _update_state(self):
        if self.check_on_set is False and self.default is not None:
            for o in self.default:
                self._ensure_value_is_in_objects(o)