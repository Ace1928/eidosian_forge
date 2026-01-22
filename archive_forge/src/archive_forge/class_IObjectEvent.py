from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IObjectEvent(Interface):
    """An event related to an object.

    The object that generated this event is not necessarily the object
    referred to by location.
    """
    object = Attribute('The subject of the event.')