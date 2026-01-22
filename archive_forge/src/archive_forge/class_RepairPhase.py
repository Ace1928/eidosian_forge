from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepairPhase(_messages.Message):
    """RepairPhase tracks the repair attempts that have been made for each
  `RepairMode` specified in the `Automation` resource.

  Fields:
    retry: Output only. Records of the retry attempts for retry repair mode.
    rollback: Output only. Rollback attempt for rollback repair mode .
  """
    retry = _messages.MessageField('RetryPhase', 1)
    rollback = _messages.MessageField('RollbackAttempt', 2)