from neutron_lib._i18n import _
from neutron_lib import exceptions
class RouterNotFound(exceptions.NotFound):
    message = _('Router %(router_id)s could not be found')