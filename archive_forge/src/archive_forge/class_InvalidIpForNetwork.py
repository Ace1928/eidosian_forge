from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidIpForNetwork(BadRequest):
    """An exception indicating an invalid IP was specified for a network.

    A specialization of the BadRequest exception indicating a specified IP
    address is invalid for a network.

    :param ip_address: The IP address that's invalid on the network.
    """
    message = _('IP address %(ip_address)s is not a valid IP for any of the subnets on the specified network.')