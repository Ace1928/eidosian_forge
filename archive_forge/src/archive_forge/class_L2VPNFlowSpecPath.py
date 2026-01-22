import logging
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class L2VPNFlowSpecPath(VpnPath):
    """Represents a way of reaching an L2VPN Flow Specification destination."""
    ROUTE_FAMILY = RF_L2VPN_FLOWSPEC
    VRF_PATH_CLASS = None
    NLRI_CLASS = FlowSpecL2VPNNLRI

    def __init__(self, *args, **kwargs):
        kwargs['nexthop'] = '0.0.0.0'
        super(L2VPNFlowSpecPath, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrfl2vpnfs import L2vpnFlowSpecPath
        self.VRF_PATH_CLASS = L2vpnFlowSpecPath
        self._nexthop = None