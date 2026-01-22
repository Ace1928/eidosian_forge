from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupInPendingState(exceptions.Conflict):
    message = _('Operation cannot be performed since associated firewall group %(firewall_id)s is in %(pending_state)s.')