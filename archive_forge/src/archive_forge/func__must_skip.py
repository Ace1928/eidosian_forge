from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _must_skip(spec, entry, is_type):
    """
    Return whether we should skip the first argument on spec's `entry`
    attribute.
    """
    if not isinstance(spec, ClassTypes):
        if entry in getattr(spec, '__dict__', {}):
            return False
        spec = spec.__class__
    if not hasattr(spec, '__mro__'):
        return is_type
    for klass in spec.__mro__:
        result = klass.__dict__.get(entry, DEFAULT)
        if result is DEFAULT:
            continue
        if isinstance(result, (staticmethod, classmethod)):
            return False
        elif isinstance(getattr(result, '__get__', None), MethodWrapperTypes):
            return is_type
        else:
            return False
    return is_type