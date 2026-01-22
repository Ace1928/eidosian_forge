from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _patch_dict(self):
    values = self.values
    in_dict = self.in_dict
    clear = self.clear
    try:
        original = in_dict.copy()
    except AttributeError:
        original = {}
        for key in in_dict:
            original[key] = in_dict[key]
    self._original = original
    if clear:
        _clear_dict(in_dict)
    try:
        in_dict.update(values)
    except AttributeError:
        for key in values:
            in_dict[key] = values[key]