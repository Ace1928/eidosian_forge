import logging
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class L2VPNFlowSpecTable(VpnTable):
    """Global table to store L2VPN Flow Specification routing information.

    Uses `L2VPNFlowSpecDest` to store destination information for each known
    Flow Specification paths.
    """
    ROUTE_FAMILY = RF_L2VPN_FLOWSPEC
    VPN_DEST_CLASS = L2VPNFlowSpecDest