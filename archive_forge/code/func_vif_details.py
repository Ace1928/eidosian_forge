import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def vif_details(self):
    """Return the vif_details describing the binding of the port.

        In the context of a host-specific operation on a distributed
        port, the vif_details property describes the binding for the
        host for which the port operation is being
        performed. Otherwise, it is the same value as
        current['binding:vif_details'].
        """