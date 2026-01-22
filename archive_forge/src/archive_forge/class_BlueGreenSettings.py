from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BlueGreenSettings(_messages.Message):
    """Settings for blue-green upgrade.

  Fields:
    nodePoolSoakDuration: Time needed after draining entire blue pool. After
      this period, blue pool will be cleaned up.
    standardRolloutPolicy: Standard policy for the blue-green upgrade.
  """
    nodePoolSoakDuration = _messages.StringField(1)
    standardRolloutPolicy = _messages.MessageField('StandardRolloutPolicy', 2)