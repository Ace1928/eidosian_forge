from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterInterfaceInUseByRoute(exceptions.InUse):
    message = _('Router interface for subnet %(subnet_id)s on router %(router_id)s cannot be deleted, as it is required by one or more routes.')