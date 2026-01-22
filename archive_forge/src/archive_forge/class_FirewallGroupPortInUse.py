from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupPortInUse(exceptions.InUse):
    message = _('Port(s) %(port_ids)s provided already associated with other firewall group(s).')