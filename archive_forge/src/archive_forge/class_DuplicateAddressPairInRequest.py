from neutron_lib._i18n import _
from neutron_lib import exceptions
class DuplicateAddressPairInRequest(exceptions.InvalidInput):
    message = _('Request contains duplicate address pair: mac_address %(mac_address)s ip_address %(ip_address)s.')