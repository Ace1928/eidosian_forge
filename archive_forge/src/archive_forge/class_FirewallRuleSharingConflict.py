from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleSharingConflict(exceptions.NotFound):
    """FWaaS exception for sharing policies

    Raised when shared policy uses unshared rules.
    """
    message = _('Operation cannot be performed since firewall policy %(firewall_policy_id)s could not find the firewall rule %(firewall_rule_id)s. Please confirm if the firewall rule exists and is shared.')