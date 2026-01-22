from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPPortOrNetworkRequired(exceptions.InvalidInput):
    message = _('Either Port ID or Network ID must be specified for Local IP create.')