from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
@implementer(IRegistered)
class Registered(RegistrationEvent):
    pass