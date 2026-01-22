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
def _init_signal_listeners(self):
    self._signal_bus.register_listener(BgpSignalBus.BGP_DEST_CHANGED, lambda _, dest: self.enqueue_for_bgp_processing(dest))
    self._signal_bus.register_listener(BgpSignalBus.BGP_VRF_REMOVED, lambda _, route_dist: self.on_vrf_removed(route_dist))
    self._signal_bus.register_listener(BgpSignalBus.BGP_VRF_ADDED, lambda _, vrf_conf: self.on_vrf_added(vrf_conf))
    self._signal_bus.register_listener(BgpSignalBus.BGP_VRF_STATS_CONFIG_CHANGED, lambda _, vrf_conf: self.on_stats_config_change(vrf_conf))