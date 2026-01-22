from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _readlines_side_effect(*args, **kwargs):
    if handle.readlines.return_value is not None:
        return handle.readlines.return_value
    return list(_state[0])