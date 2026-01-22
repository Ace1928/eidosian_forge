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
def _construct_update(self, outgoing_route):
    """Construct update message with Outgoing-routes path attribute
        appropriately cloned/copied/updated.
        """
    update = None
    path = outgoing_route.path
    pathattr_map = path.pathattr_map
    new_pathattr = []
    if path.is_withdraw:
        if isinstance(path, Ipv4Path):
            update = BGPUpdate(withdrawn_routes=[path.nlri])
            return update
        else:
            mpunreach_attr = BGPPathAttributeMpUnreachNLRI(path.route_family.afi, path.route_family.safi, [path.nlri])
            new_pathattr.append(mpunreach_attr)
    elif self.is_route_server_client:
        nlri_list = [path.nlri]
        new_pathattr.extend(pathattr_map.values())
    else:
        if self.is_route_reflector_client:
            if BGP_ATTR_TYPE_ORIGINATOR_ID not in pathattr_map:
                originator_id = path.source
                if originator_id is None:
                    originator_id = self._common_conf.router_id
                elif isinstance(path.source, Peer):
                    originator_id = path.source.ip_address
                new_pathattr.append(BGPPathAttributeOriginatorId(value=originator_id))
            cluster_lst_attr = pathattr_map.get(BGP_ATTR_TYPE_CLUSTER_LIST)
            if cluster_lst_attr:
                cluster_list = list(cluster_lst_attr.value)
                if self._common_conf.cluster_id not in cluster_list:
                    cluster_list.insert(0, self._common_conf.cluster_id)
                new_pathattr.append(BGPPathAttributeClusterList(cluster_list))
            else:
                new_pathattr.append(BGPPathAttributeClusterList([self._common_conf.cluster_id]))
        origin_attr = None
        nexthop_attr = None
        as_path_attr = None
        as4_path_attr = None
        aggregator_attr = None
        as4_aggregator_attr = None
        extcomm_attr = None
        community_attr = None
        localpref_attr = None
        pmsi_tunnel_attr = None
        unknown_opttrans_attrs = None
        nlri_list = [path.nlri]
        if path.route_family.safi in (subaddr_family.IP_FLOWSPEC, subaddr_family.VPN_FLOWSPEC):
            next_hop = []
        elif self.is_ebgp_peer():
            next_hop = self._session_next_hop(path)
            if path.is_local() and path.has_nexthop():
                next_hop = path.nexthop
        else:
            next_hop = path.nexthop
            if self._neigh_conf.is_next_hop_self or (path.is_local() and (not path.has_nexthop())):
                next_hop = self._session_next_hop(path)
                LOG.debug('using %s as a next_hop address instead of path.nexthop %s', next_hop, path.nexthop)
        nexthop_attr = BGPPathAttributeNextHop(next_hop)
        assert nexthop_attr, 'Missing NEXTHOP mandatory attribute.'
        if not isinstance(path, Ipv4Path):
            mpnlri_attr = BGPPathAttributeMpReachNLRI(path.route_family.afi, path.route_family.safi, next_hop, nlri_list)
        origin_attr = pathattr_map.get(BGP_ATTR_TYPE_ORIGIN)
        assert origin_attr, 'Missing ORIGIN mandatory attribute.'
        path_aspath = pathattr_map.get(BGP_ATTR_TYPE_AS_PATH)
        assert path_aspath, 'Missing AS_PATH mandatory attribute.'
        as_path_list = path_aspath.path_seg_list
        if not self.is_ebgp_peer():
            pass
        elif len(as_path_list) > 0 and isinstance(as_path_list[0], list) and (len(as_path_list[0]) < 255):
            as_path_list[0].insert(0, self.local_as)
        else:
            as_path_list.insert(0, [self.local_as])
        as_path_list, as4_path_list = self._trans_as_path(as_path_list)
        if self.is_four_octet_as_number_cap_valid():
            as_path_attr = BGPPathAttributeAsPath(as_path_list, as_pack_str='!I')
        else:
            as_path_attr = BGPPathAttributeAsPath(as_path_list)
        if as4_path_list:
            as4_path_attr = BGPPathAttributeAs4Path(as4_path_list)
        aggregator_attr = pathattr_map.get(BGP_ATTR_TYPE_AGGREGATOR)
        if aggregator_attr and (not self.is_four_octet_as_number_cap_valid()):
            aggregator_as_number = aggregator_attr.as_number
            if not is_valid_old_asn(aggregator_as_number):
                aggregator_attr = bgp.BGPPathAttributeAggregator(bgp.AS_TRANS, aggregator_attr.addr)
                as4_aggregator_attr = bgp.BGPPathAttributeAs4Aggregator(aggregator_as_number, aggregator_attr.addr)
        multi_exit_disc = None
        if self.is_ebgp_peer():
            if self._neigh_conf.multi_exit_disc:
                multi_exit_disc = BGPPathAttributeMultiExitDisc(self._neigh_conf.multi_exit_disc)
            else:
                pass
        if not self.is_ebgp_peer():
            multi_exit_disc = pathattr_map.get(BGP_ATTR_TYPE_MULTI_EXIT_DISC)
        if not self.is_ebgp_peer():
            localpref_attr = BGPPathAttributeLocalPref(self._common_conf.local_pref)
            key = const.ATTR_MAPS_LABEL_DEFAULT
            if isinstance(path, (Vpnv4Path, Vpnv6Path)):
                nlri = nlri_list[0]
                rf = VRF_RF_IPV4 if isinstance(path, Vpnv4Path) else VRF_RF_IPV6
                key = ':'.join([nlri.route_dist, rf])
            attr_type = AttributeMap.ATTR_LOCAL_PREF
            at_maps = self._attribute_maps.get(key, {})
            result = self._lookup_attribute_map(at_maps, attr_type, path)
            if result:
                localpref_attr = result
        community_attr = pathattr_map.get(BGP_ATTR_TYPE_COMMUNITIES)
        path_extcomm_attr = pathattr_map.get(BGP_ATTR_TYPE_EXTENDED_COMMUNITIES)
        if path_extcomm_attr:
            communities = path_extcomm_attr.communities
            if self._neigh_conf.soo_list:
                soo_list = self._neigh_conf.soo_list
                subtype = 3
                for soo in soo_list:
                    first, second = soo.split(':')
                    if '.' in first:
                        c = BGPIPv4AddressSpecificExtendedCommunity(subtype=subtype, ipv4_address=first, local_administrator=int(second))
                    else:
                        c = BGPTwoOctetAsSpecificExtendedCommunity(subtype=subtype, as_number=int(first), local_administrator=int(second))
                    communities.append(c)
            extcomm_attr = BGPPathAttributeExtendedCommunities(communities=communities)
            pmsi_tunnel_attr = pathattr_map.get(BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE)
        unknown_opttrans_attrs = bgp_utils.get_unknown_opttrans_attr(path)
        if isinstance(path, Ipv4Path):
            new_pathattr.append(nexthop_attr)
        else:
            new_pathattr.append(mpnlri_attr)
        new_pathattr.append(origin_attr)
        new_pathattr.append(as_path_attr)
        if as4_path_attr:
            new_pathattr.append(as4_path_attr)
        if aggregator_attr:
            new_pathattr.append(aggregator_attr)
        if as4_aggregator_attr:
            new_pathattr.append(as4_aggregator_attr)
        if multi_exit_disc:
            new_pathattr.append(multi_exit_disc)
        if localpref_attr:
            new_pathattr.append(localpref_attr)
        if community_attr:
            new_pathattr.append(community_attr)
        if extcomm_attr:
            new_pathattr.append(extcomm_attr)
        if pmsi_tunnel_attr:
            new_pathattr.append(pmsi_tunnel_attr)
        if unknown_opttrans_attrs:
            new_pathattr.extend(unknown_opttrans_attrs.values())
    if isinstance(path, Ipv4Path):
        update = BGPUpdate(path_attributes=new_pathattr, nlri=nlri_list)
    else:
        update = BGPUpdate(path_attributes=new_pathattr)
    return update