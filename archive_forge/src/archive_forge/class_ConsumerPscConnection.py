from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumerPscConnection(_messages.Message):
    """PSC connection details on consumer side.

  Enums:
    ErrorTypeValueValuesEnum: The error type indicates whether the error is
      consumer facing, producer facing or system internal.
    StateValueValuesEnum: The state of the PSC connection.

  Fields:
    error: The most recent error during operating this connection.
    errorInfo: Output only. The error info for the latest error during
      operating this connection.
    errorType: The error type indicates whether the error is consumer facing,
      producer facing or system internal.
    forwardingRule: The URI of the consumer forwarding rule created. Example:
      projects/{projectNumOrId}/regions/us-east1/networks/{resourceId}.
    gceOperation: The last Compute Engine operation to setup PSC connection.
    ip: The IP literal allocated on the consumer network for the PSC
      forwarding rule that is created to connect to the producer service
      attachment in this service connection map.
    network: The consumer network whose PSC forwarding rule is connected to
      the service attachments in this service connection map. Note that the
      network could be on a different project (shared VPC).
    project: The consumer project whose PSC forwarding rule is connected to
      the service attachments in this service connection map.
    pscConnectionId: The PSC connection id of the PSC forwarding rule
      connected to the service attachments in this service connection map.
    selectedSubnetwork: Output only. The URI of the selected subnetwork
      selected to allocate IP address for this connection.
    serviceAttachmentUri: The URI of a service attachment which is the target
      of the PSC connection.
    state: The state of the PSC connection.
  """

    class ErrorTypeValueValuesEnum(_messages.Enum):
        """The error type indicates whether the error is consumer facing,
    producer facing or system internal.

    Values:
      CONNECTION_ERROR_TYPE_UNSPECIFIED: An invalid error type as the default
        case.
      ERROR_INTERNAL: The error is due to Service Automation system internal.
      ERROR_CONSUMER_SIDE: The error is due to the setup on consumer side.
      ERROR_PRODUCER_SIDE: The error is due to the setup on producer side.
    """
        CONNECTION_ERROR_TYPE_UNSPECIFIED = 0
        ERROR_INTERNAL = 1
        ERROR_CONSUMER_SIDE = 2
        ERROR_PRODUCER_SIDE = 3

    class StateValueValuesEnum(_messages.Enum):
        """The state of the PSC connection.

    Values:
      STATE_UNSPECIFIED: An invalid state as the default case.
      ACTIVE: The connection is fully established and ready to use.
      FAILED: The connection is not functional since some resources on the
        connection fail to be created.
      CREATING: The connection is being created.
      DELETING: The connection is being deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        FAILED = 2
        CREATING = 3
        DELETING = 4
    error = _messages.MessageField('GoogleRpcStatus', 1)
    errorInfo = _messages.MessageField('GoogleRpcErrorInfo', 2)
    errorType = _messages.EnumField('ErrorTypeValueValuesEnum', 3)
    forwardingRule = _messages.StringField(4)
    gceOperation = _messages.StringField(5)
    ip = _messages.StringField(6)
    network = _messages.StringField(7)
    project = _messages.StringField(8)
    pscConnectionId = _messages.StringField(9)
    selectedSubnetwork = _messages.StringField(10)
    serviceAttachmentUri = _messages.StringField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)