import logging
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class EvpnPath(VpnPath):
    """Represents a way of reaching an EVPN destination."""
    ROUTE_FAMILY = RF_L2_EVPN
    VRF_PATH_CLASS = None
    NLRI_CLASS = EvpnNLRI

    def __init__(self, *args, **kwargs):
        super(EvpnPath, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrfevpn import VrfEvpnPath
        self.VRF_PATH_CLASS = VrfEvpnPath