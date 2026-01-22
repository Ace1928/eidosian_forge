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
def _unjelly_reference(self, lst):
    refid = lst[0]
    exp = lst[1]
    o = self.unjelly(exp)
    ref = self.references.get(refid)
    if ref is None:
        self.references[refid] = o
    elif isinstance(ref, NotKnown):
        ref.resolveDependants(o)
        self.references[refid] = o
    else:
        assert 0, 'Multiple references with same ID!'
    return o