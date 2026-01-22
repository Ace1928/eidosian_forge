import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackNetworkACL:
    """
    a ACL rule in the given network (the network has to belong to VPC)
    """

    def __init__(self, id, protocol, acl_id, action, cidr_list, start_port, end_port, traffic_type=None):
        """
        a ACL rule in the given network (the network has to belong to
        VPC)

        @note: This is a non-standard extension API, and only works for
               Cloudstack.

        :param      id: the ID of the ACL Item
        :type       id ``int``

        :param      protocol: the protocol for the ACL rule. Valid values are
                               TCP/UDP/ICMP/ALL or valid protocol number
        :type       protocol: ``string``

        :param      acl_id: Name of the network ACL List
        :type       acl_id: ``str``

        :param      action: scl entry action, allow or deny
        :type       action: ``string``

        :param      cidr_list: the cidr list to allow traffic from/to
        :type       cidr_list: ``str``

        :param      start_port: the starting port of ACL
        :type       start_port: ``str``

        :param      end_port: the ending port of ACL
        :type       end_port: ``str``

        :param      traffic_type: the traffic type for the ACL,can be Ingress
                                  or Egress, defaulted to Ingress if not
                                  specified
        :type       traffic_type: ``str``

        :rtype: :class:`CloudStackNetworkACL`
        """
        self.id = id
        self.protocol = protocol
        self.acl_id = acl_id
        self.action = action
        self.cidr_list = cidr_list
        self.start_port = start_port
        self.end_port = end_port
        self.traffic_type = traffic_type

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id == other.id