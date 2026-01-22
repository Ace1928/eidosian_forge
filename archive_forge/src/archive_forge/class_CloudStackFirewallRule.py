import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackFirewallRule:
    """
    A firewall rule.
    """

    def __init__(self, id, address, cidr_list, protocol, icmp_code=None, icmp_type=None, start_port=None, end_port=None):
        """
        A Firewall rule.

        @note: This is a non-standard extension API, and only works for
               CloudStack.

        :param      id: Firewall Rule ID
        :type       id: ``int``

        :param      address: External IP address
        :type       address: :class:`CloudStackAddress`

        :param      cidr_list: cidr list
        :type       cidr_list: ``str``

        :param      protocol: TCP/IP Protocol (TCP, UDP)
        :type       protocol: ``str``

        :param      icmp_code: Error code for this icmp message
        :type       icmp_code: ``int``

        :param      icmp_type: Type of the icmp message being sent
        :type       icmp_type: ``int``

        :param      start_port: start of port range
        :type       start_port: ``int``

        :param      end_port: end of port range
        :type       end_port: ``int``

        :rtype: :class:`CloudStackFirewallRule`
        """
        self.id = id
        self.address = address
        self.cidr_list = cidr_list
        self.protocol = protocol
        self.icmp_code = icmp_code
        self.icmp_type = icmp_type
        self.start_port = start_port
        self.end_port = end_port

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id == other.id