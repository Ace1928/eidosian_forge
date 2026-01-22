from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def adapter_hook(provided, object, name='', default=None):
    """Adapt an object using a registered adapter factory.

        name must be text.
        """