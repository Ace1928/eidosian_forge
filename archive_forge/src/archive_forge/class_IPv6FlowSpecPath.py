import logging
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
class IPv6FlowSpecPath(Path):
    """Represents a way of reaching an IPv6 Flow Specification destination."""
    ROUTE_FAMILY = RF_IPv6_FLOWSPEC
    VRF_PATH_CLASS = None
    NLRI_CLASS = FlowSpecIPv6NLRI

    def __init__(self, *args, **kwargs):
        kwargs['nexthop'] = '::'
        super(IPv6FlowSpecPath, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf6fs import Vrf6FlowSpecPath
        self.VRF_PATH_CLASS = Vrf6FlowSpecPath
        self._nexthop = None