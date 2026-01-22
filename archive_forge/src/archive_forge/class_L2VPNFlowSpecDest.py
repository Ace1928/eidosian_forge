import logging
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class L2VPNFlowSpecDest(VpnDest):
    """L2VPN Flow Specification Destination

    Store Flow Specification Paths.
    """
    ROUTE_FAMILY = RF_L2VPN_FLOWSPEC