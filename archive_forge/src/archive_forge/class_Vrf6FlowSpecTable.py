import logging
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.services.protocols.bgp.info_base.vpnv6fs import VPNv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecDest
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecTable
class Vrf6FlowSpecTable(VRFFlowSpecTable):
    """Virtual Routing and Forwarding information base
    for IPv6 Flow Specification.
    """
    ROUTE_FAMILY = RF_IPv6_FLOWSPEC
    VPN_ROUTE_FAMILY = RF_VPNv6_FLOWSPEC
    NLRI_CLASS = FlowSpecIPv6NLRI
    VRF_PATH_CLASS = Vrf6FlowSpecPath
    VRF_DEST_CLASS = Vrf6FlowSpecDest