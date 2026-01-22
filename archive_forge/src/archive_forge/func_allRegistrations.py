import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
def allRegistrations(self):
    """
        Yields tuples ``(required, provided, name, value)`` for all
        the registrations that this object holds.

        These tuples could be passed as the arguments to the
        :meth:`register` method on another adapter registry to
        duplicate the registrations this object holds.

        .. versionadded:: 5.3.0
        """
    yield from self._all_entries(self._adapters)