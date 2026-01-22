from oslo_utils import excutils
from neutron_lib._i18n import _
class CTZoneExhaustedError(NeutronException):
    message = _('IPtables conntrack zones exhausted, iptables rules cannot be applied.')