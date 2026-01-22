from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetAllocationError(NeutronException):
    message = _('Failed to allocate subnet: %(reason)s.')