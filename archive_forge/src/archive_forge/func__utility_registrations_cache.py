from collections import defaultdict
from zope.interface.adapter import AdapterRegistry
from zope.interface.declarations import implementedBy
from zope.interface.declarations import implementer
from zope.interface.declarations import implementer_only
from zope.interface.declarations import providedBy
from zope.interface.interface import Interface
from zope.interface.interfaces import ComponentLookupError
from zope.interface.interfaces import IAdapterRegistration
from zope.interface.interfaces import IComponents
from zope.interface.interfaces import IHandlerRegistration
from zope.interface.interfaces import ISpecification
from zope.interface.interfaces import ISubscriptionAdapterRegistration
from zope.interface.interfaces import IUtilityRegistration
from zope.interface.interfaces import Registered
from zope.interface.interfaces import Unregistered
@property
def _utility_registrations_cache(self):
    cache = self._v_utility_registrations_cache
    if cache is None or cache._utilities is not self.utilities or cache._utility_registrations is not self._utility_registrations:
        cache = self._v_utility_registrations_cache = _UtilityRegistrations(self.utilities, self._utility_registrations)
    return cache