from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RollbackTargetConfig(_messages.Message):
    """Configs for the Rollback rollout.

  Fields:
    rollout: Optional. The rollback `Rollout` to create.
    startingPhaseId: Optional. The starting phase ID for the `Rollout`. If
      unspecified, the `Rollout` will start in the stable phase.
  """
    rollout = _messages.MessageField('Rollout', 1)
    startingPhaseId = _messages.StringField(2)