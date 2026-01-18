import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
def update_local_rt_nlris(self):
    """Does book-keeping of local RT NLRIs based on all configured VRFs.

        Syncs all import RTs and RT NLRIs.
        The method should be called when any VRFs are added/removed/changed.
        """
    current_conf_import_rts = set()
    for vrf in self._vrfs_conf.vrf_confs:
        current_conf_import_rts.update(vrf.import_rts)
    removed_rts = self._all_vrfs_import_rts_set - current_conf_import_rts
    new_rts = current_conf_import_rts - self._all_vrfs_import_rts_set
    self._all_vrfs_import_rts_set = current_conf_import_rts
    for new_rt in new_rts:
        self.add_rt_nlri(new_rt)
    for removed_rt in removed_rts:
        self.add_rt_nlri(removed_rt, is_withdraw=True)