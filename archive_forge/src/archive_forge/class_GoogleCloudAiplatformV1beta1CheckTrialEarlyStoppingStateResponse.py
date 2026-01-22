from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1CheckTrialEarlyStoppingStateResponse(_messages.Message):
    """Response message for VizierService.CheckTrialEarlyStoppingState.

  Fields:
    shouldStop: True if the Trial should stop.
  """
    shouldStop = _messages.BooleanField(1)