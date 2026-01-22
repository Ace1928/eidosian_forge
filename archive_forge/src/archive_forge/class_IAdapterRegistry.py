from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
class IAdapterRegistry(Interface):
    """Provide an interface-based registry for adapters

    This registry registers objects that are in some sense "from" a
    sequence of specification to an interface and a name.

    No specific semantics are assumed for the registered objects,
    however, the most common application will be to register factories
    that adapt objects providing required specifications to a provided
    interface.
    """

    def register(required, provided, name, value):
        """Register a value

        A value is registered for a *sequence* of required specifications, a
        provided interface, and a name, which must be text.
        """

    def registered(required, provided, name=''):
        """Return the component registered for the given interfaces and name

        name must be text.

        Unlike the lookup method, this methods won't retrieve
        components registered for more specific required interfaces or
        less specific provided interfaces.

        If no component was registered exactly for the given
        interfaces and name, then None is returned.

        """

    def lookup(required, provided, name='', default=None):
        """Lookup a value

        A value is looked up based on a *sequence* of required
        specifications, a provided interface, and a name, which must be
        text.
        """

    def queryMultiAdapter(objects, provided, name='', default=None):
        """Adapt a sequence of objects to a named, provided, interface
        """

    def lookup1(required, provided, name='', default=None):
        """Lookup a value using a single required interface

        A value is looked up based on a single required
        specifications, a provided interface, and a name, which must be
        text.
        """

    def queryAdapter(object, provided, name='', default=None):
        """Adapt an object using a registered adapter factory.
        """

    def adapter_hook(provided, object, name='', default=None):
        """Adapt an object using a registered adapter factory.

        name must be text.
        """

    def lookupAll(required, provided):
        """Find all adapters from the required to the provided interfaces

        An iterable object is returned that provides name-value two-tuples.
        """

    def names(required, provided):
        """Return the names for which there are registered objects
        """

    def subscribe(required, provided, subscriber):
        """Register a subscriber

        A subscriber is registered for a *sequence* of required
        specifications, a provided interface, and a name.

        Multiple subscribers may be registered for the same (or
        equivalent) interfaces.

        .. versionchanged:: 5.1.1
           Correct the method signature to remove the ``name`` parameter.
           Subscribers have no names.
        """

    def subscribed(required, provided, subscriber):
        """
        Check whether the object *subscriber* is registered directly
        with this object via a previous call to
        ``subscribe(required, provided, subscriber)``.

        If the *subscriber*, or one equal to it, has been subscribed,
        for the given *required* sequence and *provided* interface,
        return that object. (This does not guarantee whether the *subscriber*
        itself is returned, or an object equal to it.)

        If it has not, return ``None``.

        Unlike :meth:`subscriptions`, this method won't retrieve
        components registered for more specific required interfaces or
        less specific provided interfaces.

        .. versionadded:: 5.3.0
        """

    def subscriptions(required, provided):
        """
        Get a sequence of subscribers.

        Subscribers for a sequence of *required* interfaces, and a *provided*
        interface are returned. This takes into account subscribers
        registered with this object, as well as those registered with
        base adapter registries in the resolution order, and interfaces that
        extend *provided*.

        .. versionchanged:: 5.1.1
           Correct the method signature to remove the ``name`` parameter.
           Subscribers have no names.
        """

    def subscribers(objects, provided):
        """
        Get a sequence of subscription **adapters**.

        This is like :meth:`subscriptions`, but calls the returned
        subscribers with *objects* (and optionally returns the results
        of those calls), instead of returning the subscribers directly.

        :param objects: A sequence of objects; they will be used to
            determine the *required* argument to :meth:`subscriptions`.
        :param provided: A single interface, or ``None``, to pass
            as the *provided* parameter to :meth:`subscriptions`.
            If an interface is given, the results of calling each returned
            subscriber with the the *objects* are collected and returned
            from this method; each result should be an object implementing
            the *provided* interface. If ``None``, the resulting subscribers
            are still called, but the results are ignored.
        :return: A sequence of the results of calling the subscribers
            if *provided* is not ``None``. If there are no registered
            subscribers, or *provided* is ``None``, this will be an empty
            sequence.

        .. versionchanged:: 5.1.1
           Correct the method signature to remove the ``name`` parameter.
           Subscribers have no names.
        """