import abc
from neutron_lib.api.definitions import portbindings
@property
@abc.abstractmethod
def vif_type(self):
    """Return the vif_type indicating the binding state of the port.

        In the context of a host-specific operation on a distributed
        port, the vif_type property indicates the binding state for
        the host for which the port operation is being
        performed. Otherwise, it is the same value as
        current['binding:vif_type'].
        """