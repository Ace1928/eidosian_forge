import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
def update_rtc_as_set(self):
    """Syncs RT NLRIs for new and removed RTC_ASes.

        This method should be called when a neighbor is added or removed.
        """
    curr_rtc_as_set = self._neighbors_conf.rtc_as_set
    curr_rtc_as_set.add(self._core_service.asn)
    removed_rtc_as_set = self._all_rtc_as_set - curr_rtc_as_set
    new_rtc_as_set = curr_rtc_as_set - self._all_rtc_as_set
    self._all_rtc_as_set = curr_rtc_as_set
    for new_rtc_as in new_rtc_as_set:
        for import_rt in self._all_vrfs_import_rts_set:
            self._add_rt_nlri_for_as(new_rtc_as, import_rt)
    for removed_rtc_as in removed_rtc_as_set:
        for import_rt in self._all_vrfs_import_rts_set:
            self._add_rt_nlri_for_as(removed_rtc_as, import_rt, is_withdraw=True)