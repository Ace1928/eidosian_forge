from os_ken.services.protocols.bgp.base import Activity
from os_ken.lib import hub
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
import socket
import logging
from calendar import timegm
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPPathAttributeMpUnreachNLRI
def _construct_route_monitoring(self, peer, route):
    if peer.is_mpbgp_cap_valid(bgp.RF_IPv4_VPN) or peer.is_mpbgp_cap_valid(bgp.RF_IPv6_VPN):
        peer_type = bmp.BMP_PEER_TYPE_L3VPN
    else:
        peer_type = bmp.BMP_PEER_TYPE_GLOBAL
    peer_distinguisher = 0
    peer_as = peer._neigh_conf.remote_as
    peer_bgp_id = peer.protocol.recv_open_msg.bgp_identifier
    peer_address, _ = peer.protocol._remotename
    bgp_update = self._construct_update(route.path)
    is_post_policy = not route.filtered
    timestamp = timegm(route.timestamp)
    msg = bmp.BMPRouteMonitoring(bgp_update=bgp_update, peer_type=peer_type, is_post_policy=is_post_policy, peer_distinguisher=peer_distinguisher, peer_address=peer_address, peer_as=peer_as, peer_bgp_id=peer_bgp_id, timestamp=timestamp)
    return msg