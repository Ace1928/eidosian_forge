from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def unregisterHandler(handler=None, required=None, name=''):
    """Unregister a handler.

        A handler is a subscriber that doesn't compute an adapter
        but performs some function when called.

        :returns: A boolean is returned indicating whether the registry was
            changed.

        :param handler:
            This is the object used to handle some event
            represented by the objects passed to it. The handler
            can be None, in which case any handlers registered for
            the given required specifications with the given are
            unregistered.

        :param required:
            This is a sequence of specifications for objects to be
            adapted.  If omitted, then the value of the factory's
            ``__component_adapts__`` attribute will be used.  The
            ``__component_adapts__`` attribute is
            normally set using the adapter
            decorator.  If the factory doesn't have a
            ``__component_adapts__`` adapts attribute, then this
            argument is required.

        :param name:
            The handler name.

            Currently, only the empty string is accepted.  Other
            strings will be accepted in the future when support for
            named handlers is added.

        An `IUnregistered` event is generated with an `IHandlerRegistration`.
        """