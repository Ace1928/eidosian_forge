from oslo_utils import excutils
from neutron_lib._i18n import _
class GatewayConflictWithAllocationPools(InUse):
    message = _('Gateway ip %(ip_address)s conflicts with allocation pool %(pool)s.')