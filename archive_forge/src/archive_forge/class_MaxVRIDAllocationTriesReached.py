from neutron_lib._i18n import _
from neutron_lib import exceptions
class MaxVRIDAllocationTriesReached(exceptions.NeutronException):
    message = _('Failed to allocate a VRID in the network %(network_id)s for the router %(router_id)s after %(max_tries)s tries.')