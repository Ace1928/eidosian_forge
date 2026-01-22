from neutron_lib._i18n import _
from neutron_lib import exceptions
class PortSecurityAndIPRequiredForSecurityGroups(exceptions.InvalidInput):
    message = _('Port security must be enabled and port must have an IP address in order to use security groups.')