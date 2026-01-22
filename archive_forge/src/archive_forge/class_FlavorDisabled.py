from neutron_lib._i18n import _
from neutron_lib import exceptions
class FlavorDisabled(exceptions.ServiceUnavailable):
    message = _('Flavor is not enabled.')