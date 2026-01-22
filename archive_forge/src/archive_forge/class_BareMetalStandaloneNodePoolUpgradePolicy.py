from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneNodePoolUpgradePolicy(_messages.Message):
    """BareMetalStandaloneNodePoolUpgradePolicy defines the node pool upgrade
  policy.

  Fields:
    parallelUpgradeConfig: The parallel upgrade settings for worker node
      pools.
  """
    parallelUpgradeConfig = _messages.MessageField('BareMetalStandaloneParallelUpgradeConfig', 1)