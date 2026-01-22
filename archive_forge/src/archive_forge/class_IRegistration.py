from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IRegistration(Interface):
    """A registration-information object
    """
    registry = Attribute('The registry having the registration')
    name = Attribute('The registration name')
    info = Attribute('Information about the registration\n\n    This is information deemed useful to people browsing the\n    configuration of a system. It could, for example, include\n    commentary or information about the source of the configuration.\n    ')