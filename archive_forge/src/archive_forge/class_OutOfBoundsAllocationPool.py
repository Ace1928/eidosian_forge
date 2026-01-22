from oslo_utils import excutils
from neutron_lib._i18n import _
class OutOfBoundsAllocationPool(BadRequest):
    message = _('The allocation pool %(pool)s spans beyond the subnet cidr %(subnet_cidr)s.')