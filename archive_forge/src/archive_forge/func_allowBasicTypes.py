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
def allowBasicTypes(self):
    """
        Allow all `basic' types.  (Dictionary and list.  Int, string, and float
        are implicitly allowed.)
        """
    self.allowTypes(*self.basicTypes)