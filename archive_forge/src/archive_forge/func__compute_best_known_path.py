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
def _compute_best_known_path(self):
    """Computes the best path among known paths.

        Returns current best path among `known_paths`.
        """
    if not self._known_path_list:
        from os_ken.services.protocols.bgp.processor import BgpProcessorError
        raise BgpProcessorError(desc='Need at-least one known path to compute best path')
    current_best_path = self._known_path_list[0]
    best_path_reason = BPR_ONLY_PATH
    for next_path in self._known_path_list[1:]:
        from os_ken.services.protocols.bgp.processor import compute_best_path
        new_best_path, reason = compute_best_path(self._core_service.asn, current_best_path, next_path)
        best_path_reason = reason
        if new_best_path is not None:
            current_best_path = new_best_path
    return (current_best_path, best_path_reason)