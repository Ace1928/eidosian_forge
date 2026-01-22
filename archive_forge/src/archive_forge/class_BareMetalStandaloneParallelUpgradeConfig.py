from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneParallelUpgradeConfig(_messages.Message):
    """BareMetalStandaloneParallelUpgradeConfig defines the parallel upgrade
  settings for worker node pools.

  Fields:
    concurrentNodes: The maximum number of nodes that can be upgraded at once.
    minimumAvailableNodes: The minimum number of nodes that should be healthy
      and available during an upgrade. If set to the default value of 0, it is
      possible that none of the nodes will be available during an upgrade.
  """
    concurrentNodes = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    minimumAvailableNodes = _messages.IntegerField(2, variant=_messages.Variant.INT32)