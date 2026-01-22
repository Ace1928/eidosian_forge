from neutron_lib._i18n import _
from neutron_lib import exceptions
class UnableToMatchGateways(exceptions.NeutronException):
    message = _('Unable to match a requested gateway port to existing gateway ports for router %(router_id)s')