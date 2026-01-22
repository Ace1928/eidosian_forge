import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
def _add_rt_nlri_for_as(self, rtc_as, route_target, is_withdraw=False):
    from os_ken.services.protocols.bgp.core import EXPECTED_ORIGIN
    rt_nlri = RouteTargetMembershipNLRI(rtc_as, route_target)
    pattrs = OrderedDict()
    if not is_withdraw:
        pattrs[BGP_ATTR_TYPE_ORIGIN] = BGPPathAttributeOrigin(EXPECTED_ORIGIN)
        pattrs[BGP_ATTR_TYPE_AS_PATH] = BGPPathAttributeAsPath([])
    path = RtcPath(None, rt_nlri, 0, is_withdraw=is_withdraw, pattrs=pattrs)
    tm = self._core_service.table_manager
    tm.learn_path(path)