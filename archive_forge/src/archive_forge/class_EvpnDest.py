import logging
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class EvpnDest(VpnDest):
    """EVPN Destination

    Store EVPN Paths.
    """
    ROUTE_FAMILY = RF_L2_EVPN