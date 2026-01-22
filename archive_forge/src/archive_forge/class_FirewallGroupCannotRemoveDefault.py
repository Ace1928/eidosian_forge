from neutron_lib._i18n import _
from neutron_lib import exceptions
class FirewallGroupCannotRemoveDefault(exceptions.InUse):
    message = _('Deleting default firewall group not allowed.')