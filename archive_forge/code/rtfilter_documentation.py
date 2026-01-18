import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.info_base.rtc import RtcPath
Updates interested RT list.

        Check if interested RTs have changes from previous check.
        Takes appropriate action for new interesting RTs and removal of un-
        interesting RTs.
        