from oslo_utils import excutils
from neutron_lib._i18n import _
class PortBound(InUse):
    """An operational error indicating a port is already bound.

    A specialization of the InUse exception indicating an operation can't
    complete because the port is already bound.

    :param port_id: The UUID of the port requested.
    :param vif_type: The VIF type associated with the bound port.
    :param old_mac: The old MAC address of the port.
    :param net_mac: The new MAC address of the port.
    """
    message = _('Unable to complete operation on port %(port_id)s, port is already bound, port type: %(vif_type)s, old_mac %(old_mac)s, new_mac %(new_mac)s.')