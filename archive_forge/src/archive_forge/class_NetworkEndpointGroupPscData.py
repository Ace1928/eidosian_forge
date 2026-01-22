from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkEndpointGroupPscData(_messages.Message):
    """All data that is specifically relevant to only network endpoint groups
  of type PRIVATE_SERVICE_CONNECT.

  Enums:
    PscConnectionStatusValueValuesEnum: [Output Only] The connection status of
      the PSC Forwarding Rule.

  Fields:
    consumerPscAddress: [Output Only] Address allocated from given subnetwork
      for PSC. This IP address acts as a VIP for a PSC NEG, allowing it to act
      as an endpoint in L7 PSC-XLB.
    pscConnectionId: [Output Only] The PSC connection id of the PSC Network
      Endpoint Group Consumer.
    pscConnectionStatus: [Output Only] The connection status of the PSC
      Forwarding Rule.
  """

    class PscConnectionStatusValueValuesEnum(_messages.Enum):
        """[Output Only] The connection status of the PSC Forwarding Rule.

    Values:
      ACCEPTED: The connection has been accepted by the producer.
      CLOSED: The connection has been closed by the producer and will not
        serve traffic going forward.
      NEEDS_ATTENTION: The connection has been accepted by the producer, but
        the producer needs to take further action before the forwarding rule
        can serve traffic.
      PENDING: The connection is pending acceptance by the producer.
      REJECTED: The connection has been rejected by the producer.
      STATUS_UNSPECIFIED: <no description>
    """
        ACCEPTED = 0
        CLOSED = 1
        NEEDS_ATTENTION = 2
        PENDING = 3
        REJECTED = 4
        STATUS_UNSPECIFIED = 5
    consumerPscAddress = _messages.StringField(1)
    pscConnectionId = _messages.IntegerField(2, variant=_messages.Variant.UINT64)
    pscConnectionStatus = _messages.EnumField('PscConnectionStatusValueValuesEnum', 3)