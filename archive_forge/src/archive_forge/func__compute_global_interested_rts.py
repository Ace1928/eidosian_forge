import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
def _compute_global_interested_rts(self):
    """Computes current global interested RTs for global tables.

        Computes interested RTs based on current RT filters for peers. This
        filter should be used to check if for RTs on a path that is installed
        in any global table (expect RT Table).
        """
    interested_rts = set()
    for rtfilter in self._peer_to_rtfilter_map.values():
        interested_rts.update(rtfilter)
    interested_rts.update(self._vrfs_conf.vrf_interested_rts)
    interested_rts.add(RouteTargetMembershipNLRI.DEFAULT_RT)
    interested_rts.remove(RouteTargetMembershipNLRI.DEFAULT_RT)
    return interested_rts