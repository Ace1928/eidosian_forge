from neutron_lib._i18n import _
from neutron_lib import exceptions
class MixedIPVersionsForIPSecEndpoints(exceptions.BadRequest):
    message = _('Endpoints in group %(group)s do not have the same IP version, as required for IPSec site-to-site connection')