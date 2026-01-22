import abc
from neutron_lib.api.definitions import portbindings
class NetworkContext(object, metaclass=abc.ABCMeta):
    """Context passed to MechanismDrivers for changes to network resources.

    A NetworkContext instance wraps a network resource. It provides
    helper methods for accessing other relevant information. Results
    from expensive operations are cached so that other
    MechanismDrivers can freely access the same information.
    """

    @property
    @abc.abstractmethod
    def current(self):
        """Return the network in its current configuration.

        Return the network, as defined by NeutronPluginBaseV2.
        create_network and all extensions in the ml2 plugin, with
        all its properties 'current' at the time the context was
        established.
        """

    @property
    @abc.abstractmethod
    def original(self):
        """Return the network in its original configuration.

        Return the network, with all its properties set to their
        original values prior to a call to update_network. Method is
        only valid within calls to update_network_precommit and
        update_network_postcommit.
        """

    @property
    @abc.abstractmethod
    def network_segments(self):
        """Return the segments associated with this network resource."""