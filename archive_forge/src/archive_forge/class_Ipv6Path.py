import logging
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
class Ipv6Path(Path):
    """Represents a way of reaching an v6 destination."""
    ROUTE_FAMILY = RF_IPv6_UC
    VRF_PATH_CLASS = None
    NLRI_CLASS = IPAddrPrefix

    def __init__(self, *args, **kwargs):
        super(Ipv6Path, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6Path
        self.VRF_PATH_CLASS = Vrf6Path