from neutron_lib._i18n import _
from neutron_lib import exceptions
class NonExistingSubnetInEndpointGroup(exceptions.InvalidInput):
    message = _('Subnet %(subnet)s in endpoint group does not exist')