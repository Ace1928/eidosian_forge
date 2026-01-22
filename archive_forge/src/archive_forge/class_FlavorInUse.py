from neutron_lib._i18n import _
from neutron_lib import exceptions
class FlavorInUse(exceptions.InUse):
    message = _('Flavor %(flavor_id)s is used by some service instance.')