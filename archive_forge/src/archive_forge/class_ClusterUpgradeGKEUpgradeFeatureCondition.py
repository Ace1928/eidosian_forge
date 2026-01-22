from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeGKEUpgradeFeatureCondition(_messages.Message):
    """GKEUpgradeFeatureCondition describes the condition of the feature for
  GKE clusters at a certain point of time.

  Fields:
    reason: Reason why the feature is in this status.
    status: Status of the condition, one of True, False, Unknown.
    type: Type of the condition, for example, "ready".
    updateTime: Last timestamp the condition was updated.
  """
    reason = _messages.StringField(1)
    status = _messages.StringField(2)
    type = _messages.StringField(3)
    updateTime = _messages.StringField(4)