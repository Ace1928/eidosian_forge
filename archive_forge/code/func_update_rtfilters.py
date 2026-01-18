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
def update_rtfilters(self):
    """Updates RT filters for each peer.

        Should be called if a new RT Nlri's have changed based on the setting.
        Currently only used by `Processor` to update the RT filters after it
        has processed a RT destination. If RT filter has changed for a peer we
        call RT filter change handler.
        """
    new_peer_to_rtfilter_map = self._compute_rtfilter_map()
    for peer in self._peer_manager.iterpeers:
        pre_rt_filter = self._rt_mgr.peer_to_rtfilter_map.get(peer, set())
        curr_rt_filter = new_peer_to_rtfilter_map.get(peer, set())
        old_rts = pre_rt_filter - curr_rt_filter
        new_rts = curr_rt_filter - pre_rt_filter
        if new_rts or old_rts:
            LOG.debug('RT Filter for peer %s updated: Added RTs %s, Removed Rts %s', peer.ip_address, new_rts, old_rts)
            self._on_update_rt_filter(peer, new_rts, old_rts)
    self._peer_manager.set_peer_to_rtfilter_map(new_peer_to_rtfilter_map)
    self._rt_mgr.peer_to_rtfilter_map = new_peer_to_rtfilter_map
    LOG.debug('Updated RT filters: %s', self._rt_mgr.peer_to_rtfilter_map)
    self._rt_mgr.update_interested_rts()