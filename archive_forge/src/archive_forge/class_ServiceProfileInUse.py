from neutron_lib._i18n import _
from neutron_lib import exceptions
class ServiceProfileInUse(exceptions.InUse):
    message = _('Service Profile %(sp_id)s is used by some service instance.')