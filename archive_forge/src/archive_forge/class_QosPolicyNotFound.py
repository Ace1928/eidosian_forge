from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QosPolicyNotFound(e.NotFound):
    message = _('QoS policy %(policy_id)s could not be found.')