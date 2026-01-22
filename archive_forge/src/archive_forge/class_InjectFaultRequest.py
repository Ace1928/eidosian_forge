from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InjectFaultRequest(_messages.Message):
    """Message for triggering fault injection on an instance

  Enums:
    FaultTypeValueValuesEnum: Required. The type of fault to be injected in an
      instance.

  Fields:
    faultType: Required. The type of fault to be injected in an instance.
    nodeIds: Optional. Full name of the nodes as obtained from
      INSTANCE_VIEW_FULL to subject the fault injection upon. Only applicable
      for read instances, where at least 1 node should be passed.
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes after the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    validateOnly: Optional. If set, performs request validation (e.g.
      permission checks and any other type of validation), but do not actually
      execute the fault injection.
  """

    class FaultTypeValueValuesEnum(_messages.Enum):
        """Required. The type of fault to be injected in an instance.

    Values:
      FAULT_TYPE_UNSPECIFIED: The fault type is unknown.
      STOP_VM: Stop the VM
    """
        FAULT_TYPE_UNSPECIFIED = 0
        STOP_VM = 1
    faultType = _messages.EnumField('FaultTypeValueValuesEnum', 1)
    nodeIds = _messages.StringField(2, repeated=True)
    requestId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)