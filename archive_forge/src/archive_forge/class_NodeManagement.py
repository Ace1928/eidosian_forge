from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeManagement(_messages.Message):
    """NodeManagement defines the set of node management services turned on for
  the node pool.

  Fields:
    autoRepair: A flag that specifies whether the node auto-repair is enabled
      for the node pool. If enabled, the nodes in this node pool will be
      monitored and, if they fail health checks too many times, an automatic
      repair action will be triggered.
    autoUpgrade: A flag that specifies whether node auto-upgrade is enabled
      for the node pool. If enabled, node auto-upgrade helps keep the nodes in
      your node pool up to date with the latest release version of Kubernetes.
    upgradeOptions: Specifies the Auto Upgrade knobs for the node pool.
  """
    autoRepair = _messages.BooleanField(1)
    autoUpgrade = _messages.BooleanField(2)
    upgradeOptions = _messages.MessageField('AutoUpgradeOptions', 3)