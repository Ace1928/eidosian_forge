from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkAttachmentConnectedEndpoint(_messages.Message):
    """[Output Only] A connection connected to this network attachment.

  Enums:
    StatusValueValuesEnum: The status of a connected endpoint to this network
      attachment.

  Fields:
    ipAddress: The IPv4 address assigned to the producer instance network
      interface. This value will be a range in case of Serverless.
    ipv6Address: The IPv6 address assigned to the producer instance network
      interface. This is only assigned when the stack types of both the
      instance network interface and the consumer subnet are IPv4_IPv6.
    projectIdOrNum: The project id or number of the interface to which the IP
      was assigned.
    secondaryIpCidrRanges: Alias IP ranges from the same subnetwork.
    status: The status of a connected endpoint to this network attachment.
    subnetwork: The subnetwork used to assign the IP to the producer instance
      network interface.
    subnetworkCidrRange: [Output Only] The CIDR range of the subnet from which
      the IPv4 internal IP was allocated from.
  """

    class StatusValueValuesEnum(_messages.Enum):
        """The status of a connected endpoint to this network attachment.

    Values:
      ACCEPTED: The consumer allows traffic from the producer to reach its
        VPC.
      CLOSED: The consumer network attachment no longer exists.
      NEEDS_ATTENTION: The consumer needs to take further action before
        traffic can be served.
      PENDING: The consumer neither allows nor prohibits traffic from the
        producer to reach its VPC.
      REJECTED: The consumer prohibits traffic from the producer to reach its
        VPC.
      STATUS_UNSPECIFIED: <no description>
    """
        ACCEPTED = 0
        CLOSED = 1
        NEEDS_ATTENTION = 2
        PENDING = 3
        REJECTED = 4
        STATUS_UNSPECIFIED = 5
    ipAddress = _messages.StringField(1)
    ipv6Address = _messages.StringField(2)
    projectIdOrNum = _messages.StringField(3)
    secondaryIpCidrRanges = _messages.StringField(4, repeated=True)
    status = _messages.EnumField('StatusValueValuesEnum', 5)
    subnetwork = _messages.StringField(6)
    subnetworkCidrRange = _messages.StringField(7)