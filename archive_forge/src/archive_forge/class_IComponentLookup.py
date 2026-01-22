from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IComponentLookup(Interface):
    """Component Manager for a Site

    This object manages the components registered at a particular site. The
    definition of a site is intentionally vague.
    """
    adapters = Attribute('Adapter Registry to manage all registered adapters.')
    utilities = Attribute('Adapter Registry to manage all registered utilities.')

    def queryAdapter(object, interface, name='', default=None):
        """Look for a named adapter to an interface for an object

        If a matching adapter cannot be found, returns the default.
        """

    def getAdapter(object, interface, name=''):
        """Look for a named adapter to an interface for an object

        If a matching adapter cannot be found, a `ComponentLookupError`
        is raised.
        """

    def queryMultiAdapter(objects, interface, name='', default=None):
        """Look for a multi-adapter to an interface for multiple objects

        If a matching adapter cannot be found, returns the default.
        """

    def getMultiAdapter(objects, interface, name=''):
        """Look for a multi-adapter to an interface for multiple objects

        If a matching adapter cannot be found, a `ComponentLookupError`
        is raised.
        """

    def getAdapters(objects, provided):
        """Look for all matching adapters to a provided interface for objects

        Return an iterable of name-adapter pairs for adapters that
        provide the given interface.
        """

    def subscribers(objects, provided):
        """Get subscribers

        Subscribers are returned that provide the provided interface
        and that depend on and are computed from the sequence of
        required objects.
        """

    def handle(*objects):
        """Call handlers for the given objects

        Handlers registered for the given objects are called.
        """

    def queryUtility(interface, name='', default=None):
        """Look up a utility that provides an interface.

        If one is not found, returns default.
        """

    def getUtilitiesFor(interface):
        """Look up the registered utilities that provide an interface.

        Returns an iterable of name-utility pairs.
        """

    def getAllUtilitiesRegisteredFor(interface):
        """Return all registered utilities for an interface

        This includes overridden utilities.

        An iterable of utility instances is returned.  No names are
        returned.
        """