from oslo_utils import excutils
from neutron_lib._i18n import _
class ExternalIpAddressExhausted(BadRequest):
    """An error due to not finding IP addresses on an external network.

    A specialization of the BadRequest exception indicating no IP addresses
    can be found on a network.

    :param net_id: The UUID of the network.
    """
    message = _('Unable to find any IP address on external network %(net_id)s.')