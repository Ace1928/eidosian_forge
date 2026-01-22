from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IHandlerRegistration(IRegistration):
    handler = Attribute('An object called used to handle an event')
    required = Attribute('The handled interfaces\n\n    This is a sequence of interfaces handled by the registered\n    handler.  The handler will be caled with a sequence of objects, as\n    positional arguments, that provide these interfaces.\n    ')