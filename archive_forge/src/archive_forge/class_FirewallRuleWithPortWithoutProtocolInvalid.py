from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleWithPortWithoutProtocolInvalid(exceptions.InvalidInput):
    message = _('Source/destination port requires a protocol.')