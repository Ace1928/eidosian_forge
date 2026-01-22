from oslo_utils import excutils
from neutron_lib._i18n import _
class IllegalSubnetPoolAssociationToAddressScope(BadRequest):
    message = _('Illegal subnetpool association: subnetpool %(subnetpool_id)s cannot be associated with address scope %(address_scope_id)s.')