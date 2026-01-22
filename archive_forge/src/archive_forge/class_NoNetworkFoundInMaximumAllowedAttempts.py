from oslo_utils import excutils
from neutron_lib._i18n import _
class NoNetworkFoundInMaximumAllowedAttempts(ServiceUnavailable):
    message = _('Unable to create the network. No available network found in maximum allowed attempts.')