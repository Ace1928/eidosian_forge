from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class PortQosBindingNotFound(e.NotFound):
    message = _('QoS binding for port %(port_id)s and policy %(policy_id)s could not be found.')