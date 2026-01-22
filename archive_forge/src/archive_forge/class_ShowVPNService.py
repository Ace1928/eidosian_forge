from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronv20
class ShowVPNService(neutronv20.ShowCommand):
    """Show information of a given VPN service."""
    resource = 'vpnservice'
    help_resource = 'VPN service'