from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _instance_callable(obj):
    """Given an object, return True if the object is callable.
    For classes, return True if instances would be callable."""
    if not isinstance(obj, ClassTypes):
        return getattr(obj, '__call__', None) is not None
    if six.PY3:
        for base in (obj,) + obj.__mro__:
            if base.__dict__.get('__call__') is not None:
                return True
    else:
        klass = obj
        if klass.__dict__.get('__call__') is not None:
            return True
        for base in klass.__bases__:
            if _instance_callable(base):
                return True
    return False