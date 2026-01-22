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
class Vrf4Table(VrfTable):
    """Virtual Routing and Forwarding information base for IPv4."""
    ROUTE_FAMILY = RF_IPv4_UC
    VPN_ROUTE_FAMILY = RF_IPv4_VPN
    NLRI_CLASS = IPAddrPrefix
    VRF_PATH_CLASS = Vrf4Path
    VRF_DEST_CLASS = Vrf4Dest