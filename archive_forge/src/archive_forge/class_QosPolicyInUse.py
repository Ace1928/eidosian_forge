from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QosPolicyInUse(e.InUse):
    message = _('QoS Policy %(policy_id)s is used by %(object_type)s %(object_id)s.')