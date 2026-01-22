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
def _genericUnjelly(self, cls, state):
    """
        Unjelly a type for which no specific unjellier is registered, but which
        is nonetheless allowed.

        @param cls: the class of the instance we are unjellying.
        @type cls: L{type}

        @param state: The jellied representation of the object's state; its
            C{__dict__} unless it has a C{__setstate__} that takes something
            else.
        @type state: L{list}

        @return: the new, unjellied instance.
        """
    return self._maybePostUnjelly(_newInstance(cls, self.unjelly(state)))