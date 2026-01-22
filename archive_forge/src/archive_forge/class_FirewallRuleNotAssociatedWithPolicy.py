from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleNotAssociatedWithPolicy(exceptions.InvalidInput):
    message = _('Firewall rule %(firewall_rule_id)s is not associated with firewall policy %(firewall_policy_id)s.')