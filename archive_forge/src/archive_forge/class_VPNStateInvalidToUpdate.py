from neutron_lib._i18n import _
from neutron_lib import exceptions
class VPNStateInvalidToUpdate(exceptions.BadRequest):
    message = _('Invalid state %(state)s of vpnaas resource %(id)s for updating')