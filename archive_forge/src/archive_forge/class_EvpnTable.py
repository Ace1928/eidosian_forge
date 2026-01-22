import logging
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class EvpnTable(VpnTable):
    """Global table to store EVPN routing information.

    Uses `EvpnDest` to store destination information for each known EVPN
    paths.
    """
    ROUTE_FAMILY = RF_L2_EVPN
    VPN_DEST_CLASS = EvpnDest