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
def _cmp_by_router_id(local_asn, path1, path2):
    """Select the route received from the peer with the lowest BGP router ID.

    If both paths are eBGP paths, then we do not do any tie breaking, i.e we do
    not pick best-path based on this criteria.
    RFC: http://tools.ietf.org/html/rfc5004
    We pick best path between two iBGP paths as usual.
    """

    def get_asn(path_source):
        if path_source is None:
            return local_asn
        else:
            return path_source.remote_as

    def get_router_id(path, local_bgp_id):
        path_source = path.source
        if path_source is None:
            return local_bgp_id
        else:
            originator_id = path.get_pattr(BGP_ATTR_TYPE_ORIGINATOR_ID)
            if originator_id:
                return originator_id.value
            return path_source.protocol.recv_open_msg.bgp_identifier
    path_source1 = path1.source
    path_source2 = path2.source
    if path_source1 is None and path_source2 is None:
        return None
    asn1 = get_asn(path_source1)
    asn2 = get_asn(path_source2)
    is_ebgp1 = asn1 != local_asn
    is_ebgp2 = asn2 != local_asn
    if is_ebgp1 and is_ebgp2:
        return None
    if is_ebgp1 is True and is_ebgp2 is False or (is_ebgp1 is False and is_ebgp2 is True):
        raise ValueError('This method does not support comparing ebgp with ibgp path')
    if path_source1 is not None:
        local_bgp_id = path_source1.protocol.sent_open_msg.bgp_identifier
    else:
        local_bgp_id = path_source2.protocol.sent_open_msg.bgp_identifier
    router_id1 = get_router_id(path1, local_bgp_id)
    router_id2 = get_router_id(path2, local_bgp_id)
    if router_id1 == router_id2:
        return None
    from os_ken.services.protocols.bgp.utils.bgp import from_inet_ptoi
    if from_inet_ptoi(router_id1) < from_inet_ptoi(router_id2):
        return path1
    else:
        return path2