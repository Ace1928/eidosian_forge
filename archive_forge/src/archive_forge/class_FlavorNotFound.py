from neutron_lib._i18n import _
from neutron_lib import exceptions
class FlavorNotFound(exceptions.NotFound):
    message = _('Flavor %(flavor_id)s could not be found.')