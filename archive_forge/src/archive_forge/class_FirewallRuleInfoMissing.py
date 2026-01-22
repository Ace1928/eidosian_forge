from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallRuleInfoMissing(exceptions.InvalidInput):
    message = _('Missing rule info argument for insert/remove rule operation.')