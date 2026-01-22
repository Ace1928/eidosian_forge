from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QosRuleNotFound(e.NotFound):
    message = _('QoS rule %(rule_id)s for policy %(policy_id)s could not be found.')