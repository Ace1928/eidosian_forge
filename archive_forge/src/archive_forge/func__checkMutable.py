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
def _checkMutable(self, obj):
    objId = id(obj)
    if objId in self.cooked:
        return self.cooked[objId]
    if objId in self.preserved:
        self._cook(obj)
        return self.cooked[objId]