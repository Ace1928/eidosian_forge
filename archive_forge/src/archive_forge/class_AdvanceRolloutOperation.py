from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdvanceRolloutOperation(_messages.Message):
    """Contains the information of an automated advance-rollout operation.

  Fields:
    destinationPhase: Output only. The phase the rollout will be advanced to.
    rollout: Output only. The name of the rollout that initiates the
      `AutomationRun`.
    sourcePhase: Output only. The phase of a deployment that initiated the
      operation.
    wait: Output only. How long the operation will be paused.
  """
    destinationPhase = _messages.StringField(1)
    rollout = _messages.StringField(2)
    sourcePhase = _messages.StringField(3)
    wait = _messages.StringField(4)