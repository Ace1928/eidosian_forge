from oslo_utils import excutils
from neutron_lib._i18n import _
class PhysicalNetworkNameError(NeutronException):
    message = _('Empty physical network name.')