from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedRolloutConfig(_messages.Message):
    """The configuration used for the Rollout. Waves are assigned
  automatically.

  Fields:
    soakDuration: Optional. Soak time before starting the next wave.
  """
    soakDuration = _messages.StringField(1)