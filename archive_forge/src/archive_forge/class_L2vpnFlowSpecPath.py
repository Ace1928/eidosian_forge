import logging
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.services.protocols.bgp.info_base.l2vpnfs import L2VPNFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecDest
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecTable
class L2vpnFlowSpecPath(VRFFlowSpecPath):
    """Represents a way of reaching an IP destination with
    a L2VPN Flow Specification.
    """
    ROUTE_FAMILY = RF_L2VPN_FLOWSPEC
    VPN_PATH_CLASS = L2VPNFlowSpecPath
    VPN_NLRI_CLASS = FlowSpecL2VPNNLRI