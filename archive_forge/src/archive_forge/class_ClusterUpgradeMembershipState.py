from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeMembershipState(_messages.Message):
    """Per-membership state for this feature.

  Fields:
    ignored: Whether this membership is ignored by the feature. For example,
      manually upgraded clusters can be ignored if they are newer than the
      default versions of its release channel.
    upgrades: Actual upgrade state against desired.
  """
    ignored = _messages.MessageField('ClusterUpgradeIgnoredMembership', 1)
    upgrades = _messages.MessageField('ClusterUpgradeMembershipGKEUpgradeState', 2, repeated=True)