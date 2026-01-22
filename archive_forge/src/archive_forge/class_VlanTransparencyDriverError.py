from neutron_lib._i18n import _
from neutron_lib import exceptions
class VlanTransparencyDriverError(exceptions.NeutronException):
    """Vlan Transparency not supported by all mechanism drivers."""
    message = _('Backend does not support VLAN Transparency.')