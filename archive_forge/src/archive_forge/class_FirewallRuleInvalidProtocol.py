from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleInvalidProtocol(exceptions.InvalidInput):
    message = _('Protocol %(protocol)s is not supported. Only protocol values %(values)s and their integer representation (0 to 255) are supported.')