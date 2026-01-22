from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IAttribute(IElement):
    """Attribute descriptors"""
    interface = Attribute('interface', 'Stores the interface instance in which the attribute is located.')