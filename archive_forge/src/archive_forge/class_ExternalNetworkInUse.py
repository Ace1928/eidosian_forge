from neutron_lib._i18n import _
from neutron_lib import exceptions
class ExternalNetworkInUse(exceptions.InUse):
    message = _('External network %(net_id)s cannot be updated to be made non-external, since it has existing gateway ports.')