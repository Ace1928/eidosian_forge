from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackTargetResponse(_messages.Message):
    """The response object from `RollbackTarget`.

  Fields:
    rollbackConfig: The config of the rollback `Rollout` created or will be
      created.
  """
    rollbackConfig = _messages.MessageField('RollbackTargetConfig', 1)