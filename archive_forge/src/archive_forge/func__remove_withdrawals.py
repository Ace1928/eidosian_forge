import abc
import logging
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import EvpnEthernetSegmentNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGPEncapsulationExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsiLabelExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsImportRTExtendedCommunity
from os_ken.lib.packet.bgp import BGPPathAttributePmsiTunnel
from os_ken.lib.packet.bgp import PmsiTunnelIdIngressReplication
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnIpPrefixNLRI
from os_ken.lib.packet.safi import (
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.utils.bgp import create_rt_extended_community
from os_ken.services.protocols.bgp.utils.stats import LOCAL_ROUTES
from os_ken.services.protocols.bgp.utils.stats import REMOTE_ROUTES
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_ID
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_NAME
def _remove_withdrawals(self):
    """Removes withdrawn paths.

        Note:
        We may have disproportionate number of withdraws compared to know paths
        since not all paths get installed into the table due to bgp policy and
        we can receive withdraws for such paths and withdrawals may not be
        stopped by the same policies.
        """
    LOG.debug('Removing %s withdrawals', len(self._withdraw_list))
    if not self._withdraw_list:
        return
    if not self._known_path_list:
        LOG.debug('Found %s withdrawals for path(s) that did not get installed.', len(self._withdraw_list))
        del self._withdraw_list[:]
        return
    matches = []
    w_matches = []
    for withdraw in self._withdraw_list:
        match = None
        for path in self._known_path_list:
            if path.puid == withdraw.puid:
                match = path
                matches.append(path)
                w_matches.append(withdraw)
                break
        if not match:
            LOG.debug('No matching path for withdraw found, may be path was not installed into table: %s', withdraw)
    if len(matches) != len(self._withdraw_list):
        LOG.debug('Did not find match for some withdrawals. Number of matches(%s), number of withdrawals (%s)', len(matches), len(self._withdraw_list))
    for match in matches:
        self._known_path_list.remove(match)
    for w_match in w_matches:
        self._withdraw_list.remove(w_match)