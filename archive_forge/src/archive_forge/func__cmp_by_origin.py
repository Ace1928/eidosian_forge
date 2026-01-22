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
def _cmp_by_origin(path1, path2):
    """Select the best path based on origin attribute.

    IGP is preferred over EGP; EGP is preferred over Incomplete.
    If both paths have same origin, we return None.
    """

    def get_origin_pref(origin):
        if origin.value == BGP_ATTR_ORIGIN_IGP:
            return 3
        elif origin.value == BGP_ATTR_ORIGIN_EGP:
            return 2
        elif origin.value == BGP_ATTR_ORIGIN_INCOMPLETE:
            return 1
        else:
            LOG.error('Invalid origin value encountered %s.', origin)
            return 0
    origin1 = path1.get_pattr(BGP_ATTR_TYPE_ORIGIN)
    origin2 = path2.get_pattr(BGP_ATTR_TYPE_ORIGIN)
    assert origin1 is not None and origin2 is not None
    if origin1.value == origin2.value:
        return None
    origin1 = get_origin_pref(origin1)
    origin2 = get_origin_pref(origin2)
    if origin1 == origin2:
        return None
    elif origin1 > origin2:
        return path1
    return path2