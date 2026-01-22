from oslo_utils import excutils
from neutron_lib._i18n import _
class TunnelIdInUse(InUse):
    """A network creation failure due to tunnel ID already in use.

    A specialization of the InUse exception indicating network creation failed
    because a said tunnel ID is already in use.

    :param tunnel_id: The ID of the tunnel that's already in use.
    """
    message = _('Unable to create the network. The tunnel ID %(tunnel_id)s is in use.')