import logging
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGP_PROCESSOR_ERROR_CODE
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.utils import circlist
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_LOCAL_PREF
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_EGP
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
def _cmp_by_cluster_list(path1, path2):
    """Selects the route received from the peer with the shorter
    CLUSTER_LIST length. [RFC4456]

    The CLUSTER_LIST length is evaluated as zero if a route does not
    carry the CLUSTER_LIST attribute.
    """

    def _get_cluster_list_len(path):
        c_list = path.get_pattr(BGP_ATTR_TYPE_CLUSTER_LIST)
        if c_list is None:
            return 0
        else:
            return len(c_list.value)
    c_list_len1 = _get_cluster_list_len(path1)
    c_list_len2 = _get_cluster_list_len(path2)
    if c_list_len1 < c_list_len2:
        return path1
    elif c_list_len1 > c_list_len2:
        return path2
    else:
        return None