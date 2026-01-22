from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
@implementer(IRegistrationEvent)
class RegistrationEvent(ObjectEvent):
    """There has been a change in a registration
    """

    def __repr__(self):
        return '{} event:\n{!r}'.format(self.__class__.__name__, self.object)