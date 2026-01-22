from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressScopeUpdateError(exceptions.BadRequest):
    message = _('Unable to update address scope %(address_scope_id)s : %(reason)s.')