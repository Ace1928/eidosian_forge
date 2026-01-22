from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomationRolloutMetadata(_messages.Message):
    """AutomationRolloutMetadata contains Automation-related actions that were
  performed on a rollout.

  Fields:
    advanceAutomationRuns: Output only. The IDs of the AutomationRuns
      initiated by an advance rollout rule.
    currentRepairAutomationRun: Output only. The current AutomationRun
      repairing the rollout.
    promoteAutomationRun: Output only. The ID of the AutomationRun initiated
      by a promote release rule.
    repairAutomationRuns: Output only. The IDs of the AutomationRuns initiated
      by a repair rollout rule.
  """
    advanceAutomationRuns = _messages.StringField(1, repeated=True)
    currentRepairAutomationRun = _messages.StringField(2)
    promoteAutomationRun = _messages.StringField(3)
    repairAutomationRuns = _messages.StringField(4, repeated=True)