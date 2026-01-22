import itertools
import weakref
from zope.interface import Interface
from zope.interface import implementer
from zope.interface import providedBy
from zope.interface import ro
from zope.interface._compat import _normalize_name
from zope.interface._compat import _use_c_impl
from zope.interface.interfaces import IAdapterRegistry
@implementer(IAdapterRegistry)
class VerifyingAdapterRegistry(BaseAdapterRegistry):
    """
    The most commonly-used adapter registry.
    """
    LookupClass = VerifyingAdapterLookup