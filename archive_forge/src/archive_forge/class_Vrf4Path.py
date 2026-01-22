import logging
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import LabelledVPNIPAddrPrefix
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vrf import VrfDest
from os_ken.services.protocols.bgp.info_base.vrf import VrfNlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf import VrfPath
from os_ken.services.protocols.bgp.info_base.vrf import VrfTable
class Vrf4Path(VrfPath):
    """Represents a way of reaching an IP destination with a VPN."""
    ROUTE_FAMILY = RF_IPv4_UC
    VPN_PATH_CLASS = Vpnv4Path
    VPN_NLRI_CLASS = LabelledVPNIPAddrPrefix