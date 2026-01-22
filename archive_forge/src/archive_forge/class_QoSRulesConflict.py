from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QoSRulesConflict(e.Conflict):
    message = _('Rule %(new_rule_type)s conflicts with rule %(rule_id)s which already exists in QoS Policy %(policy_id)s.')