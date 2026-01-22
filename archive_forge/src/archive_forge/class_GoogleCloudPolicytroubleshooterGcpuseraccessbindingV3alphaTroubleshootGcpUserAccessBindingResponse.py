from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaTroubleshootGcpUserAccessBindingResponse(_messages.Message):
    """Response for troubleshooting GcpUserAccessBinding.

  Enums:
    AccessStateValueValuesEnum: Output only. The access state of the request.

  Fields:
    accessState: Output only. The access state of the request.
    gcpUserAccessBindingExplanations: The explanation of the
      GcpUserAccessBinding.
    principal: The principal email address of the caller.
  """

    class AccessStateValueValuesEnum(_messages.Enum):
        """Output only. The access state of the request.

    Values:
      ACCESS_STATE_UNSPECIFIED: Not used
      ACCESS_STATE_GRANTED: The request is granted by GcpUserAccessBinding.
      ACCESS_STATE_DENIED: The request is denied by GcpUserAccessBinding.
      ACCESS_STATE_NOT_APPLICABLE: GcpUserAccessBinding are not applicable to
        principal.
      ACCESS_STATE_UNKNOWN: No enough information to get a conclusion.
    """
        ACCESS_STATE_UNSPECIFIED = 0
        ACCESS_STATE_GRANTED = 1
        ACCESS_STATE_DENIED = 2
        ACCESS_STATE_NOT_APPLICABLE = 3
        ACCESS_STATE_UNKNOWN = 4
    accessState = _messages.EnumField('AccessStateValueValuesEnum', 1)
    gcpUserAccessBindingExplanations = _messages.MessageField('GoogleCloudPolicytroubleshooterGcpuseraccessbindingV3alphaGcpUserAccessBindingExplanation', 2, repeated=True)
    principal = _messages.StringField(3)