from neutron_lib._i18n import _
from neutron_lib import exceptions
class VPNPeerAddressNotResolved(exceptions.InvalidInput):
    message = _('Peer address %(peer_address)s cannot be resolved')