from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QoSRuleParameterConflict(e.Conflict):
    message = _('Unable to add the rule with value %(rule_value)s to the policy %(policy_id)s as the existing rule of type %(existing_rule)s has value %(existing_value)s.')