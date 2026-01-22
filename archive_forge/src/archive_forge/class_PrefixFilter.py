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
@functools.total_ordering
class PrefixFilter(Filter):
    """
    Used to specify a prefix for filter.

    We can create PrefixFilter object as follows::

        prefix_filter = PrefixFilter('10.5.111.0/24',
                                     policy=PrefixFilter.POLICY_PERMIT)

    ================ ==================================================
    Attribute        Description
    ================ ==================================================
    prefix           A prefix used for this filter
    policy           One of the following values.

                     | PrefixFilter.POLICY.PERMIT
                     | PrefixFilter.POLICY_DENY
    ge               Prefix length that will be applied to this filter.
                     ge means greater than or equal.
    le               Prefix length that will be applied to this filter.
                     le means less than or equal.
    ================ ==================================================

    For example, when PrefixFilter object is created as follows::

        p = PrefixFilter('10.5.111.0/24',
                         policy=PrefixFilter.POLICY_DENY,
                         ge=26, le=28)

    Prefixes which match 10.5.111.0/24 and its length matches
    from 26 to 28 will be filtered.
    When this filter is used as an out-filter, it will stop sending
    the path to neighbor because of POLICY_DENY.
    When this filter is used as in-filter, it will stop importing the path
    to the global rib because of POLICY_DENY.
    If you specify POLICY_PERMIT, the path is sent to neighbor or imported to
    the global rib.

    If you don't want to send prefixes 10.5.111.64/26 and 10.5.111.32/27
    and 10.5.111.16/28, and allow to send other 10.5.111.0's prefixes,
    you can do it by specifying as follows::

        p = PrefixFilter('10.5.111.0/24',
                         policy=PrefixFilter.POLICY_DENY,
                         ge=26, le=28).
    """

    def __init__(self, prefix, policy, ge=None, le=None):
        super(PrefixFilter, self).__init__(policy)
        self._prefix = prefix
        self._network = netaddr.IPNetwork(prefix)
        self._ge = ge
        self._le = le

    def __lt__(self, other):
        return self._network < other._network

    def __eq__(self, other):
        return self._network == other._network

    def __repr__(self):
        policy = 'PERMIT' if self._policy == self.POLICY_PERMIT else 'DENY'
        return 'PrefixFilter(prefix=%s,policy=%s,ge=%s,le=%s)' % (self._prefix, policy, self._ge, self._le)

    @property
    def prefix(self):
        return self._prefix

    @property
    def policy(self):
        return self._policy

    @property
    def ge(self):
        return self._ge

    @property
    def le(self):
        return self._le

    def evaluate(self, path):
        """ This method evaluates the prefix.

        Returns this object's policy and the result of matching.
        If the specified prefix matches this object's prefix and
        ge and le condition,
        this method returns True as the matching result.

        ``path`` specifies the path that has prefix.
        """
        nlri = path.nlri
        result = False
        length = nlri.length
        net = netaddr.IPNetwork(nlri.prefix)
        if net in self._network:
            if self._ge is None and self._le is None:
                result = True
            elif self._ge is None and self._le:
                if length <= self._le:
                    result = True
            elif self._ge and self._le is None:
                if self._ge <= length:
                    result = True
            elif self._ge and self._le:
                if self._ge <= length <= self._le:
                    result = True
        return (self.policy, result)

    def clone(self):
        """ This method clones PrefixFilter object.

        Returns PrefixFilter object that has the same values with the
        original one.
        """
        return self.__class__(self.prefix, policy=self._policy, ge=self._ge, le=self._le)