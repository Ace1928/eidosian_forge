from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReportExecutionEventRequest(_messages.Message):
    """Request message for NotebookInternalService.ReportExecutionEvent.

  Enums:
    EventTypeValueValuesEnum: Required. The type of the event.

  Fields:
    eventType: Required. The type of the event.
    status: Optional. The error details of the event.
    vmToken: Required. The VM identity token (a JWT) for authenticating the
      VM. https://cloud.google.com/compute/docs/instances/verifying-instance-
      identity
  """

    class EventTypeValueValuesEnum(_messages.Enum):
        """Required. The type of the event.

    Values:
      EVENT_TYPE_UNSPECIFIED: Unspecified.
      ACTIVE: Notebook execution process has started. Expect this message
        within expected time to provision compute.
      DONE: Notebook execution process is completed. Expect this message
        within timeout.
      FAILED: Notebook execution process has failed. Expect this message
        within timeout.
    """
        EVENT_TYPE_UNSPECIFIED = 0
        ACTIVE = 1
        DONE = 2
        FAILED = 3
    eventType = _messages.EnumField('EventTypeValueValuesEnum', 1)
    status = _messages.MessageField('GoogleRpcStatus', 2)
    vmToken = _messages.StringField(3)