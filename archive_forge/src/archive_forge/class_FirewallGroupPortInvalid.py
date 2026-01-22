from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupPortInvalid(exceptions.Conflict):
    message = _('Port %(port_id)s of firewall group is invalid.')