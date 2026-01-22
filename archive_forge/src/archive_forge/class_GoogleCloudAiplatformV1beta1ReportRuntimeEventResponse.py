from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ReportRuntimeEventResponse(_messages.Message):
    """Response message for NotebookInternalService.ReportRuntimeEvent.

  Fields:
    idleShutdownMessage: If the idle shutdown is blocked by CP, CP will send
      the block message. Otherwise, this field is not set.
  """
    idleShutdownMessage = _messages.StringField(1)