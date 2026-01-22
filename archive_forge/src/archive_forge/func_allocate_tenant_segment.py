import abc
from neutron_lib.api.definitions import portbindings
@abc.abstractmethod
def allocate_tenant_segment(self, context, filters=None):
    """Allocate resource for a new tenant network segment.

        :param context: instance of neutron context with DB session
        :param filters: a dictionary that is used as search criteria
        :returns: segment dictionary using keys defined above

        Called inside transaction context on session to allocate a new
        tenant network, typically from a type-specific resource
        pool. If successful, return a segment dictionary describing
        the segment. If tenant network segment cannot be allocated
        (i.e. tenant networks not supported or resource pool is
        exhausted), return None.
        """