from zope.interface.declarations import implementer
from zope.interface.interface import Attribute
from zope.interface.interface import Interface
def unregisterUtility(component=None, provided=None, name='', factory=None):
    """Unregister a utility

        :returns:
            A boolean is returned indicating whether the registry was
            changed.  If the given *component* is None and there is no
            component registered, or if the given *component* is not
            None and is not registered, then the function returns
            False, otherwise it returns True.

        :param factory:
           Factory for the component to be unregistered.

        :param component:
           The registered component The given component can be
           None, in which case any component registered to provide
           the given provided interface with the given name is
           unregistered.

        :param provided:
           This is the interface provided by the utility.  If the
           component is not None and provides a single interface,
           then this argument is optional and the
           component-implemented interface will be used.

        :param name:
           The utility name.

        Only one of *component* and *factory* can be used.
        An `IUnregistered` event is generated with an `IUtilityRegistration`.
        """