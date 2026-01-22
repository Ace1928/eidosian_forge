import logging
import traceback
from os_ken.lib.packet.bgp import RouteFamily
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
def _path_to_dict(dst, path):
    path_seg_list = path.get_pattr(BGP_ATTR_TYPE_AS_PATH).path_seg_list
    if isinstance(path_seg_list, list):
        aspath = []
        for as_path_seg in path_seg_list:
            for as_num in as_path_seg:
                aspath.append(as_num)
    else:
        aspath = ''
    origin = path.get_pattr(BGP_ATTR_TYPE_ORIGIN)
    origin = origin.value if origin else None
    if origin == BGP_ATTR_ORIGIN_IGP:
        origin = 'i'
    elif origin == BGP_ATTR_ORIGIN_EGP:
        origin = 'e'
    elif origin == BGP_ATTR_ORIGIN_INCOMPLETE:
        origin = '?'
    nexthop = path.nexthop
    med = path.get_pattr(BGP_ATTR_TYPE_MULTI_EXIT_DISC)
    med = med.value if med else ''
    bpr = dst.best_path_reason if path == dst.best_path else ''
    localpref = path.get_pattr(BGP_ATTR_TYPE_LOCAL_PREF)
    localpref = localpref.value if localpref else ''
    if hasattr(path.nlri, 'label_list'):
        labels = path.nlri.label_list
    else:
        labels = None
    return {'best': path == dst.best_path, 'bpr': bpr, 'prefix': path.nlri_str, 'labels': labels, 'nexthop': nexthop, 'metric': med, 'aspath': aspath, 'origin': origin, 'localpref': localpref}