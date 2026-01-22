import copy
import datetime
import decimal
import types
import warnings
from functools import reduce
from zope.interface import implementer
from incremental import Version
from twisted.persisted.crefutil import (
from twisted.python.compat import nativeString
from twisted.python.deprecate import deprecatedModuleAttribute
from twisted.python.reflect import namedAny, namedObject, qual
from twisted.spread.interfaces import IJellyable, IUnjellyable
def _newInstance(cls, state):
    """
    Make a new instance of a class without calling its __init__ method.

    @param state: A C{dict} used to update C{inst.__dict__} either directly or
        via C{__setstate__}, if available.

    @return: A new instance of C{cls}.
    """
    instance = _createBlank(cls)

    def defaultSetter(state):
        if isinstance(state, dict):
            instance.__dict__ = state or {}
    setter = getattr(instance, '__setstate__', defaultSetter)
    setter(state)
    return instance