import logging
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import LabelledVPNIP6AddrPrefix
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.info_base.vrf import VrfDest
from os_ken.services.protocols.bgp.info_base.vrf import VrfNlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf import VrfPath
from os_ken.services.protocols.bgp.info_base.vrf import VrfTable
class Vrf6Table(VrfTable):
    """Virtual Routing and Forwarding information base for IPv6."""
    ROUTE_FAMILY = RF_IPv6_UC
    VPN_ROUTE_FAMILY = RF_IPv6_VPN
    NLRI_CLASS = IP6AddrPrefix
    VRF_PATH_CLASS = Vrf6Path
    VRF_DEST_CLASS = Vrf6Dest