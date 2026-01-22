from neutron_lib._i18n import _
from neutron_lib import exceptions
class FlavorServiceProfileBindingExists(exceptions.Conflict):
    message = _('Service Profile %(sp_id)s is already associated with flavor %(fl_id)s.')