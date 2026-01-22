from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TargetPoolInstanceHealth(_messages.Message):
    """A TargetPoolInstanceHealth object.

  Fields:
    healthStatus: A HealthStatus attribute.
    kind: [Output Only] Type of resource. Always
      compute#targetPoolInstanceHealth when checking the health of an
      instance.
  """
    healthStatus = _messages.MessageField('HealthStatus', 1, repeated=True)
    kind = _messages.StringField(2, default='compute#targetPoolInstanceHealth')