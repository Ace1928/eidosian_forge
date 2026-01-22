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
def _process_paths(self):
    """Calculates best-path among known paths for this destination.

        Returns:
         - Best path

        Modifies destination's state related to stored paths. Removes withdrawn
        paths from known paths. Also, adds new paths to known paths.
        """
    self._remove_withdrawals()
    if not self._known_path_list and len(self._new_path_list) == 1:
        self._known_path_list.append(self._new_path_list[0])
        del self._new_path_list[0]
        return (self._known_path_list[0], BPR_ONLY_PATH)
    self._remove_old_paths()
    self._known_path_list.extend(self._new_path_list)
    del self._new_path_list[:]
    if not self._known_path_list:
        return (None, BPR_UNKNOWN)
    current_best_path, reason = self._compute_best_known_path()
    return (current_best_path, reason)