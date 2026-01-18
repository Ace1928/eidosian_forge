import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def update_network_segment_range_allocations(self):
    """Update driver network segment range allocations.

        This syncs the driver segment allocations when network segment ranges
        have been created, updated or deleted.
        """