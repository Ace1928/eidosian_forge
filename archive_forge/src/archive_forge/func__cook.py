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
def _cook(self, object):
    """
        (internal) Backreference an object.

        Notes on this method for the hapless future maintainer: If I've already
        gone through the prepare/preserve cycle on the specified object (it is
        being referenced after the serializer is "done with" it, e.g. this
        reference is NOT circular), the copy-in-place of aList is relevant,
        since the list being modified is the actual, pre-existing jelly
        expression that was returned for that object. If not, it's technically
        superfluous, since the value in self.preserved didn't need to be set,
        but the invariant that self.preserved[id(object)] is a list is
        convenient because that means we don't have to test and create it or
        not create it here, creating fewer code-paths.  that's why
        self.preserved is always set to a list.

        Sorry that this code is so hard to follow, but Python objects are
        tricky to persist correctly. -glyph
        """
    aList = self.preserved[id(object)]
    newList = copy.copy(aList)
    refid = self._ref_id
    self._ref_id = self._ref_id + 1
    aList[:] = [reference_atom, refid, newList]
    self.cooked[id(object)] = [dereference_atom, refid]
    return aList