import logging
from time import strftime
from os_ken.services.protocols.bgp.operator.command import Command
from os_ken.services.protocols.bgp.operator.command import CommandsResponse
from os_ken.services.protocols.bgp.operator.command import STATUS_ERROR
from os_ken.services.protocols.bgp.operator.command import STATUS_OK
from os_ken.services.protocols.bgp.operator.commands.responses import \
from os_ken.services.protocols.bgp.operator.views.bgp import CoreServiceDetailView
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
def _retrieve_paths(self, addr_family, route_family, ip_addr):
    peer_view = self._retrieve_peer_view(ip_addr)
    adj_rib_in = peer_view.c_rel('adj_rib_in')
    adj_rib_in.apply_filter(lambda k, v: addr_family == 'all' or v.path.route_family == route_family)
    return adj_rib_in