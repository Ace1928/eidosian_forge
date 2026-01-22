import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class VPNv6FlowSpecPath(VpnPath):
    """Represents a way of reaching an VPNv6 Flow Specification destination."""
    ROUTE_FAMILY = RF_VPNv6_FLOWSPEC
    VRF_PATH_CLASS = None
    NLRI_CLASS = FlowSpecVPNv6NLRI

    def __init__(self, *args, **kwargs):
        kwargs['nexthop'] = '::'
        super(VPNv6FlowSpecPath, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf6fs import Vrf6FlowSpecPath
        self.VRF_PATH_CLASS = Vrf6FlowSpecPath
        self._nexthop = None