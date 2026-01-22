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
def _maybePostUnjelly(self, unjellied):
    """
        If the given object has support for the C{postUnjelly} hook, set it up
        to be called at the end of deserialization.

        @param unjellied: an object that has already been unjellied.

        @return: C{unjellied}
        """
    if hasattr(unjellied, 'postUnjelly'):
        self.postCallbacks.append(unjellied.postUnjelly)
    return unjellied