import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class VPNv4FlowSpecDest(VpnDest):
    """VPNv4 Flow Specification Destination

    Store Flow Specification Paths.
    """
    ROUTE_FAMILY = RF_VPNv4_FLOWSPEC