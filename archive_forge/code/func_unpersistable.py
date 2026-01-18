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
def unpersistable(self, reason, sxp=None):
    """
        (internal) Returns an sexp: (unpersistable "reason").  Utility method
        for making note that a particular object could not be serialized.
        """
    if sxp is None:
        sxp = []
    sxp.append(unpersistable_atom)
    if isinstance(reason, str):
        reason = reason.encode('utf-8')
    sxp.append(reason)
    return sxp