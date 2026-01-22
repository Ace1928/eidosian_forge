from neutron_lib._i18n import _
from neutron_lib import exceptions
class MissingRequiredEndpointGroup(exceptions.BadRequest):
    message = _('Missing endpoint group%(suffix)s %(which)s for IPSec site-to-site connection')