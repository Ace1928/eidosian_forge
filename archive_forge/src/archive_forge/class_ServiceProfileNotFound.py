from neutron_lib._i18n import _
from neutron_lib import exceptions
class ServiceProfileNotFound(exceptions.NotFound):
    message = _('Service Profile %(sp_id)s could not be found.')