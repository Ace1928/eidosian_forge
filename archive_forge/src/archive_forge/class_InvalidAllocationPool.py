from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidAllocationPool(BadRequest):
    message = _('The allocation pool %(pool)s is not valid.')