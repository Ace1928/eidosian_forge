from neutron_lib._i18n import _
from neutron_lib import exceptions
class RoutesExhausted(exceptions.BadRequest):
    message = _('Unable to complete operation for %(router_id)s. The number of routes exceeds the maximum %(quota)s.')