from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallIpAddressConflict(exceptions.InvalidInput):
    message = _('Invalid input - IP addresses do not agree with IP Version.')