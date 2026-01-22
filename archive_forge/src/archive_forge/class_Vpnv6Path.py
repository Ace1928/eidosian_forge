import logging
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class Vpnv6Path(VpnPath):
    """Represents a way of reaching an VPNv4 destination."""
    ROUTE_FAMILY = RF_IPv6_VPN
    VRF_PATH_CLASS = None
    NLRI_CLASS = IP6AddrPrefix

    def __init__(self, *args, **kwargs):
        super(Vpnv6Path, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6Path
        self.VRF_PATH_CLASS = Vrf6Path