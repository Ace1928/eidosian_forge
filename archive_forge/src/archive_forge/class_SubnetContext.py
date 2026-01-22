import abc
from neutron_lib.api.definitions import portbindings
class SubnetContext(object, metaclass=abc.ABCMeta):
    """Context passed to MechanismDrivers for changes to subnet resources.

    A SubnetContext instance wraps a subnet resource. It provides
    helper methods for accessing other relevant information. Results
    from expensive operations are cached so that other
    MechanismDrivers can freely access the same information.
    """

    @property
    @abc.abstractmethod
    def current(self):
        """Return the subnet in its current configuration.

        Return the subnet, as defined by NeutronPluginBaseV2.
        create_subnet and all extensions in the ml2 plugin, with
        all its properties 'current' at the time the context was
        established.
        """

    @property
    @abc.abstractmethod
    def original(self):
        """Return the subnet in its original configuration.

        Return the subnet, with all its properties set to their
        original values prior to a call to update_subnet. Method is
        only valid within calls to update_subnet_precommit and
        update_subnet_postcommit.
        """