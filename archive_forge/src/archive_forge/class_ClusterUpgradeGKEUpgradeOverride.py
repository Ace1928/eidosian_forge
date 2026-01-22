from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeGKEUpgradeOverride(_messages.Message):
    """Properties of a GKE upgrade that can be overridden by the user. For
  example, a user can skip soaking by overriding the soaking to 0.

  Fields:
    postConditions: Required. Post conditions to override for the specified
      upgrade (name + version). Required.
    upgrade: Required. Which upgrade to override. Required.
  """
    postConditions = _messages.MessageField('ClusterUpgradePostConditions', 1)
    upgrade = _messages.MessageField('ClusterUpgradeGKEUpgrade', 2)