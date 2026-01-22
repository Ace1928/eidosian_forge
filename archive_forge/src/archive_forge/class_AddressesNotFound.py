from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressesNotFound(exceptions.NotFound):
    message = _('Addresses %(addresses)s not found in the address group %(address_group_id)s.')