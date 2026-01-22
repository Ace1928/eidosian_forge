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
class AttributeMap(object):
    """
    This class is used to specify an attribute to add if the path matches
    filters.
    We can create AttributeMap object as follows::

        pref_filter = PrefixFilter('192.168.103.0/30',
                                   PrefixFilter.POLICY_PERMIT)

        attribute_map = AttributeMap([pref_filter],
                                     AttributeMap.ATTR_LOCAL_PREF, 250)

        speaker.attribute_map_set('192.168.50.102', [attribute_map])

    AttributeMap.ATTR_LOCAL_PREF means that 250 is set as a
    local preference value if nlri in the path matches pref_filter.

    ASPathFilter is also available as a filter. ASPathFilter checks if AS_PATH
    attribute in the path matches AS number in the filter.

    =================== ==================================================
    Attribute           Description
    =================== ==================================================
    filters             A list of filter.
                        Each object should be a Filter class or its sub-class
    attr_type           A type of attribute to map on filters. Currently
                        AttributeMap.ATTR_LOCAL_PREF is available.
    attr_value          A attribute value
    =================== ==================================================
    """
    ATTR_LOCAL_PREF = '_local_pref'

    def __init__(self, filters, attr_type, attr_value):
        assert all((isinstance(f, Filter) for f in filters)), 'all the items in filters must be an instance of Filter sub-class'
        self.filters = filters
        self.attr_type = attr_type
        self.attr_value = attr_value

    def evaluate(self, path):
        """ This method evaluates attributes of the path.

        Returns the cause and result of matching.
        Both cause and result are returned from filters
        that this object contains.

        ``path`` specifies the path.
        """
        result = False
        cause = None
        for f in self.filters:
            cause, result = f.evaluate(path)
            if not result:
                break
        return (cause, result)

    def get_attribute(self):
        func = getattr(self, 'get' + self.attr_type)
        return func()

    def get_local_pref(self):
        local_pref_attr = BGPPathAttributeLocalPref(value=self.attr_value)
        return local_pref_attr

    def clone(self):
        """ This method clones AttributeMap object.

        Returns AttributeMap object that has the same values with the
        original one.
        """
        cloned_filters = [f.clone() for f in self.filters]
        return self.__class__(cloned_filters, self.attr_type, self.attr_value)

    def __repr__(self):
        attr_type = 'LOCAL_PREF' if self.attr_type == self.ATTR_LOCAL_PREF else None
        filter_string = ','.join((repr(f) for f in self.filters))
        return 'AttributeMap(filters=[%s],attribute_type=%s,attribute_value=%s)' % (filter_string, attr_type, self.attr_value)