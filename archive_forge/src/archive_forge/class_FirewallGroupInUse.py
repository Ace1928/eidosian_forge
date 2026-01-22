from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupInUse(exceptions.InUse):
    message = _('Firewall group %(firewall_id)s is still active.')