from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IComponents(IComponentLookup, IComponentRegistry):
    """Component registration and access
    """