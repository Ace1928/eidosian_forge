import abc
from abc import ABCMeta
from abc import abstractmethod
from copy import copy
import logging
import functools
import netaddr
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.processor import BPR_ONLY_PATH
from os_ken.services.protocols.bgp.processor import BPR_UNKNOWN
def withdraw_if_sent_to(self, peer):
    """Sends a withdraw for this destination to given `peer`.

        Check the records if we indeed advertise this destination to given peer
        and if so, creates a withdraw for advertised route and sends it to the
        peer.
        Parameter:
            - `peer`: (Peer) peer to send withdraw to
        """
    from os_ken.services.protocols.bgp.peer import Peer
    if not isinstance(peer, Peer):
        raise TypeError('Currently we only support sending withdrawal to instance of peer')
    sent_route = self._sent_routes.pop(peer, None)
    if not sent_route:
        return False
    sent_path = sent_route.path
    withdraw_clone = sent_path.clone(for_withdrawal=True)
    outgoing_route = OutgoingRoute(withdraw_clone)
    sent_route.sent_peer.enque_outgoing_msg(outgoing_route)
    return True