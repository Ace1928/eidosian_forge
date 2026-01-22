import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def _createLookup(self):
    self._v_lookup = self.LookupClass(self)
    for name in self._delegated:
        self.__dict__[name] = getattr(self._v_lookup, name)