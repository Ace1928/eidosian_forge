import base64
import warnings
from libcloud.utils.py3 import b, urlparse
from libcloud.compute.base import (
from libcloud.compute.types import (
from libcloud.utils.networking import is_private_subnet
from libcloud.common.cloudstack import CloudStackDriverMixIn
from libcloud.compute.providers import Provider
class CloudStackPortForwardingRule:
    """
    A Port forwarding rule for Source NAT.
    """

    def __init__(self, node, rule_id, address, protocol, public_port, private_port, public_end_port=None, private_end_port=None, network_id=None):
        """
        A Port forwarding rule for Source NAT.

        @note: This is a non-standard extension API, and only works for EC2.

        :param      node: Node for rule
        :type       node: :class:`Node`

        :param      rule_id: Rule ID
        :type       rule_id: ``int``

        :param      address: External IP address
        :type       address: :class:`CloudStackAddress`

        :param      protocol: TCP/IP Protocol (TCP, UDP)
        :type       protocol: ``str``

        :param      public_port: External port for rule (or start port if
                                 public_end_port is also provided)
        :type       public_port: ``int``

        :param      private_port: Internal node port for rule (or start port if
                                  public_end_port is also provided)
        :type       private_port: ``int``

        :param      public_end_port: End of external port range
        :type       public_end_port: ``int``

        :param      private_end_port: End of internal port range
        :type       private_end_port: ``int``

        :param      network_id: The network of the vm the Port Forwarding rule
                                will be created for. Required when public Ip
                                address is not associated with any Guest
                                network yet (VPC case)
        :type       network_id: ``str``

        :rtype: :class:`CloudStackPortForwardingRule`
        """
        self.node = node
        self.id = rule_id
        self.address = address
        self.protocol = protocol
        self.public_port = public_port
        self.public_end_port = public_end_port
        self.private_port = private_port
        self.private_end_port = private_end_port

    def delete(self):
        self.node.ex_delete_port_forwarding_rule(rule=self)

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self.id == other.id