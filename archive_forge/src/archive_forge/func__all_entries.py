import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def _all_entries(self, byorder):
    for i, components in enumerate(byorder):
        for key, value in self._allKeys(components, i + 1):
            assert len(key) == i + 2
            required = key[:i]
            provided = key[-2]
            name = key[-1]
            yield (required, provided, name, value)