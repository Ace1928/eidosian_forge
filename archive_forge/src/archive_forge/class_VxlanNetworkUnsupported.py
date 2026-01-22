from oslo_utils import excutils
from neutron_lib._i18n import _
class VxlanNetworkUnsupported(NeutronException):
    message = _('VXLAN network unsupported.')