import logging
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class Vpnv4Path(VpnPath):
    """Represents a way of reaching an VPNv4 destination."""
    ROUTE_FAMILY = RF_IPv4_VPN
    VRF_PATH_CLASS = None
    NLRI_CLASS = IPAddrPrefix

    def __init__(self, *args, **kwargs):
        super(Vpnv4Path, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4Path
        self.VRF_PATH_CLASS = Vrf4Path