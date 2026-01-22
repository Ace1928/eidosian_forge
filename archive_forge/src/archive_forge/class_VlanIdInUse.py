from oslo_utils import excutils
from neutron_lib._i18n import _
class VlanIdInUse(InUse):
    """A network operational error indicating a VLAN ID is already in use.

    A specialization of the InUse exception indicating network creation failed
    because a specified VLAN is already in use on the physical network.

    :param vlan_id: The VLAN ID.
    :param physical_network: The physical network.
    """
    message = _('Unable to create the network. The VLAN %(vlan_id)s on physical network %(physical_network)s is in use.')