from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _get_signature_object(func, as_instance, eat_self):
    """
    Given an arbitrary, possibly callable object, try to create a suitable
    signature object.
    Return a (reduced func, signature) tuple, or None.
    """
    if isinstance(func, ClassTypes) and (not as_instance):
        try:
            func = func.__init__
        except AttributeError:
            return None
        eat_self = True
    elif not isinstance(func, FunctionTypes):
        try:
            func = func.__call__
        except AttributeError:
            return None
    if eat_self:
        sig_func = partial(func, None)
    else:
        sig_func = func
    try:
        return (func, inspectsignature(sig_func))
    except ValueError:
        return None