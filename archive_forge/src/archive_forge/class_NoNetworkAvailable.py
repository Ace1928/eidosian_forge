from oslo_utils import excutils
from neutron_lib._i18n import _
class NoNetworkAvailable(ResourceExhausted):
    """A failure to create a network due to no tenant networks for allocation.

    A specialization of the ResourceExhausted exception indicating network
    creation failed because no tenant network are available for allocation.
    """
    message = _('Unable to create the network. No tenant network is available for allocation.')