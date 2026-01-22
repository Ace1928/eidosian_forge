from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicysimulatorV1alphaGenerateOrgPolicyViolationsPreviewOperationMetadata(_messages.Message):
    """GenerateOrgPolicyViolationsPreviewOperationMetadata is metadata about an
  OrgPolicyViolationsPreview generations operation.

  Enums:
    StateValueValuesEnum: The current state of the operation.

  Fields:
    requestTime: Time when the request was received.
    resourcesFound: Total number of resources that need scanning. Should equal
      resource_scanned + resources_pending
    resourcesPending: Number of resources still to scan.
    resourcesScanned: Number of resources already scanned.
    startTime: Time when the request started processing, i.e. when the state
      was set to RUNNING.
    state: The current state of the operation.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the operation.

    Values:
      PREVIEW_STATE_UNSPECIFIED: The state is unspecified.
      PREVIEW_PENDING: The OrgPolicyViolationsPreview has not been created
        yet.
      PREVIEW_RUNNING: The OrgPolicyViolationsPreview is currently being
        created.
      PREVIEW_SUCCEEDED: The OrgPolicyViolationsPreview creation finished
        successfully.
      PREVIEW_FAILED: The OrgPolicyViolationsPreview creation failed with an
        error.
    """
        PREVIEW_STATE_UNSPECIFIED = 0
        PREVIEW_PENDING = 1
        PREVIEW_RUNNING = 2
        PREVIEW_SUCCEEDED = 3
        PREVIEW_FAILED = 4
    requestTime = _messages.StringField(1)
    resourcesFound = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    resourcesPending = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    resourcesScanned = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(5)
    state = _messages.EnumField('StateValueValuesEnum', 6)