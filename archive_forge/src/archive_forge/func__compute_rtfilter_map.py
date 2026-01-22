import logging
import netaddr
import socket
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_RESET
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import STD_BGP_SERVER_PORT_NUM
from os_ken.services.protocols.bgp import core_managers
from os_ken.services.protocols.bgp.model import FlexinetOutgoingRoute
from os_ken.services.protocols.bgp.protocol import Factory
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.utils.rtfilter import RouteTargetManager
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_ACTIVE
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.bmp import BMPClient
from os_ken.lib import sockopt
from os_ken.lib import ip
def _compute_rtfilter_map(self):
    """Returns neighbor's RT filter (permit/allow filter based on RT).

        Walks RT filter tree and computes current RT filters for each peer that
        have advertised RT NLRIs.
        Returns:
            dict of peer, and `set` of rts that a particular neighbor is
            interested in.
        """
    rtfilter_map = {}

    def get_neigh_filter(neigh):
        neigh_filter = rtfilter_map.get(neigh)
        if neigh_filter is None:
            neigh_filter = set()
            rtfilter_map[neigh] = neigh_filter
        return neigh_filter
    if self._common_config.max_path_ext_rtfilter_all:
        for rtcdest in self._table_manager.get_rtc_table().values():
            known_path_list = rtcdest.known_path_list
            for path in known_path_list:
                neigh = path.source
                if neigh is None:
                    continue
                neigh_filter = get_neigh_filter(neigh)
                neigh_filter.add(path.nlri.route_target)
    else:
        for rtcdest in self._table_manager.get_rtc_table().values():
            path = rtcdest.best_path
            if not path:
                continue
            neigh = path.source
            if neigh and neigh.is_ebgp_peer():
                neigh_filter = get_neigh_filter(neigh)
                neigh_filter.add(path.nlri.route_target)
            else:
                known_path_list = rtcdest.known_path_list
                for path in known_path_list:
                    neigh = path.source
                    if neigh and (not neigh.is_ebgp_peer()):
                        neigh_filter = get_neigh_filter(neigh)
                        neigh_filter.add(path.nlri.route_target)
    return rtfilter_map