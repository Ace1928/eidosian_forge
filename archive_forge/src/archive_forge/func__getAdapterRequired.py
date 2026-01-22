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
def _getAdapterRequired(factory, required):
    if required is None:
        try:
            required = factory.__component_adapts__
        except AttributeError:
            raise TypeError("The adapter factory doesn't have a __component_adapts__ attribute and no required specifications were specified")
    elif ISpecification.providedBy(required):
        raise TypeError('the required argument should be a list of interfaces, not a single interface')
    result = []
    for r in required:
        if r is None:
            r = Interface
        elif not ISpecification.providedBy(r):
            if isinstance(r, type):
                r = implementedBy(r)
            else:
                raise TypeError('Required specification must be a specification or class, not %r' % type(r))
        result.append(r)
    return tuple(result)