from oslo_utils import excutils
from neutron_lib._i18n import _
class InvalidInputSubnetServiceType(InvalidInput):
    message = _('Subnet service type %(service_type)s is not a string.')