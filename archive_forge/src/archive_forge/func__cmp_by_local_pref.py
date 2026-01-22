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
def _cmp_by_local_pref(path1, path2):
    """Selects a path with highest local-preference.

    Unlike the weight attribute, which is only relevant to the local
    router, local preference is an attribute that routers exchange in the
    same AS. Highest local-pref is preferred. If we cannot decide,
    we return None.
    """
    lp1 = path1.get_pattr(BGP_ATTR_TYPE_LOCAL_PREF)
    lp2 = path2.get_pattr(BGP_ATTR_TYPE_LOCAL_PREF)
    if not (lp1 and lp2):
        return None
    lp1 = lp1.value
    lp2 = lp2.value
    if lp1 > lp2:
        return path1
    elif lp2 > lp1:
        return path2
    else:
        return None