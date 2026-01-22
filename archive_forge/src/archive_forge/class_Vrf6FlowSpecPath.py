import logging
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.services.protocols.bgp.info_base.vpnv6fs import VPNv6FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecDest
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecTable
class Vrf6FlowSpecPath(VRFFlowSpecPath):
    """Represents a way of reaching an IP destination with
    a VPN Flow Specification.
    """
    ROUTE_FAMILY = RF_IPv6_FLOWSPEC
    VPN_PATH_CLASS = VPNv6FlowSpecPath
    VPN_NLRI_CLASS = FlowSpecVPNv6NLRI