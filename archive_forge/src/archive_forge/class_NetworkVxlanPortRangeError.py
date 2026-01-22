from oslo_utils import excutils
from neutron_lib._i18n import _
class NetworkVxlanPortRangeError(NeutronException):
    message = _("Invalid network VXLAN port range: '%(vxlan_range)s'.")