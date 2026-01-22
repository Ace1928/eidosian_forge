from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleInUse(exceptions.InUse):
    message = _('Firewall rule %(firewall_rule_id)s is being used.')