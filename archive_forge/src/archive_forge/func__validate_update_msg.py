from collections import namedtuple
import itertools
import logging
import socket
import time
import traceback
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import Sink
from os_ken.services.protocols.bgp.base import Source
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp import constants as const
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.model import SentRoute
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
from os_ken.services.protocols.bgp.info_base.base import AttributeMap
from os_ken.services.protocols.bgp.model import ReceivedRoute
from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_PASSIVE
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4, VRF_RF_IPV6
from os_ken.services.protocols.bgp.utils import bgp as bgp_utils
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.utils.validation import is_valid_old_asn
from os_ken.lib.packet import bgp
from os_ken.lib.packet.bgp import RouteFamily
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import get_rf
from os_ken.lib.packet.bgp import BGPOpen
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPRouteRefresh
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_ADMINISTRATIVE_SHUTDOWN
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import BGP_MSG_UPDATE
from os_ken.lib.packet.bgp import BGP_MSG_KEEPALIVE
from os_ken.lib.packet.bgp import BGP_MSG_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGPPathAttributeNextHop
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeAs4Path
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeOriginatorId
from os_ken.lib.packet.bgp import BGPPathAttributeClusterList
from os_ken.lib.packet.bgp import BGPPathAttributeMpReachNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeMpUnreachNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AGGREGATOR
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS4_AGGREGATOR
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS4_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_NEXT_HOP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MP_REACH_NLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MP_UNREACH_NLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGPTwoOctetAsSpecificExtendedCommunity
from os_ken.lib.packet.bgp import BGPIPv4AddressSpecificExtendedCommunity
from os_ken.lib.packet import safi as subaddr_family
def _validate_update_msg(self, update_msg):
    """Validate update message as per RFC.

        Here we validate the message after it has been parsed. Message
        has already been validated against some errors inside parsing
        library.
        """
    assert update_msg.type == BGP_MSG_UPDATE
    if self.state.bgp_state != const.BGP_FSM_ESTABLISHED:
        LOG.error('Received UPDATE message when not in ESTABLISHED state.')
        raise bgp.FiniteStateMachineError()
    mp_reach_attr = update_msg.get_path_attr(BGP_ATTR_TYPE_MP_REACH_NLRI)
    mp_unreach_attr = update_msg.get_path_attr(BGP_ATTR_TYPE_MP_UNREACH_NLRI)
    if not (mp_reach_attr or mp_unreach_attr):
        if not self.is_mpbgp_cap_valid(RF_IPv4_UC):
            LOG.error('Got UPDATE message with un-available afi/safi %s', RF_IPv4_UC)
        nlri_list = update_msg.nlri
        if len(nlri_list) > 0:
            aspath = update_msg.get_path_attr(BGP_ATTR_TYPE_AS_PATH)
            if not aspath:
                raise bgp.MissingWellKnown(BGP_ATTR_TYPE_AS_PATH)
            if self.check_first_as and self.is_ebgp_peer() and (not aspath.has_matching_leftmost(self.remote_as)):
                LOG.error('First AS check fails. Raise appropriate exception.')
                raise bgp.MalformedAsPath()
            origin = update_msg.get_path_attr(BGP_ATTR_TYPE_ORIGIN)
            if not origin:
                raise bgp.MissingWellKnown(BGP_ATTR_TYPE_ORIGIN)
            nexthop = update_msg.get_path_attr(BGP_ATTR_TYPE_NEXT_HOP)
            if not nexthop:
                raise bgp.MissingWellKnown(BGP_ATTR_TYPE_NEXT_HOP)
        return True
    if mp_unreach_attr:
        if not self.is_mpbgp_cap_valid(mp_unreach_attr.route_family):
            LOG.error('Got UPDATE message with un-available afi/safi for MP_UNREACH path attribute (non-negotiated afi/safi) %s', mp_unreach_attr.route_family)
    if mp_reach_attr:
        if not self.is_mpbgp_cap_valid(mp_reach_attr.route_family):
            LOG.error('Got UPDATE message with un-available afi/safi for MP_UNREACH path attribute (non-negotiated afi/safi) %s', mp_reach_attr.route_family)
        aspath = update_msg.get_path_attr(BGP_ATTR_TYPE_AS_PATH)
        if not aspath:
            raise bgp.MissingWellKnown(BGP_ATTR_TYPE_AS_PATH)
        if self.check_first_as and self.is_ebgp_peer() and (not aspath.has_matching_leftmost(self.remote_as)):
            LOG.error('First AS check fails. Raise appropriate exception.')
            raise bgp.MalformedAsPath()
        origin = update_msg.get_path_attr(BGP_ATTR_TYPE_ORIGIN)
        if not origin:
            raise bgp.MissingWellKnown(BGP_ATTR_TYPE_ORIGIN)
        if mp_reach_attr.route_family.safi in (subaddr_family.IP_FLOWSPEC, subaddr_family.VPN_FLOWSPEC):
            pass
        elif not mp_reach_attr.next_hop or mp_reach_attr.next_hop == self.host_bind_ip:
            LOG.error('Nexthop of received UPDATE msg. (%s) same as local interface address %s.', mp_reach_attr.next_hop, self.host_bind_ip)
            return False
    return True