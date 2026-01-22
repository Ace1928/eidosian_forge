from oslo_utils import excutils
from neutron_lib._i18n import _
class DNSNameServersExhausted(BadRequest):
    message = _('Unable to complete operation for %(subnet_id)s. The number of DNS nameservers exceeds the limit %(quota)s.')