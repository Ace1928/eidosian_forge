from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeMembershipGKEUpgradeState(_messages.Message):
    """ScopeGKEUpgradeState is a GKEUpgrade and its state per-membership.

  Fields:
    status: Status of the upgrade.
    upgrade: Which upgrade to track the state.
  """
    status = _messages.MessageField('ClusterUpgradeUpgradeStatus', 1)
    upgrade = _messages.MessageField('ClusterUpgradeGKEUpgrade', 2)