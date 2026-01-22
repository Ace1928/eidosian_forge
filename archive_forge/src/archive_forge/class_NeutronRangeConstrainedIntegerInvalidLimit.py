from oslo_utils import reflection
from neutron_lib._i18n import _
from neutron_lib import exceptions
class NeutronRangeConstrainedIntegerInvalidLimit(exceptions.NeutronException):
    message = _('Incorrect range limits specified: start = %(start)s, end = %(end)s')