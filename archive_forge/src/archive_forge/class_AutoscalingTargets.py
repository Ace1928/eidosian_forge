from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingTargets(_messages.Message):
    """The autoscaling targets for an instance.

  Fields:
    highPriorityCpuUtilizationPercent: Required. The target high priority cpu
      utilization percentage that the autoscaler should be trying to achieve
      for the instance. This number is on a scale from 0 (no utilization) to
      100 (full utilization). The valid range is [10, 90] inclusive.
    storageUtilizationPercent: Required. The target storage utilization
      percentage that the autoscaler should be trying to achieve for the
      instance. This number is on a scale from 0 (no utilization) to 100 (full
      utilization). The valid range is [10, 99] inclusive.
  """
    highPriorityCpuUtilizationPercent = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    storageUtilizationPercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)