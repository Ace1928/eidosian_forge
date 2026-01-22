from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingLimits(_messages.Message):
    """The autoscaling limits for the instance. Users can define the minimum
  and maximum compute capacity allocated to the instance, and the autoscaler
  will only scale within that range. Users can either use nodes or processing
  units to specify the limits, but should use the same unit to set both the
  min_limit and max_limit.

  Fields:
    maxNodes: Maximum number of nodes allocated to the instance. If set, this
      number should be greater than or equal to min_nodes.
    maxProcessingUnits: Maximum number of processing units allocated to the
      instance. If set, this number should be multiples of 1000 and be greater
      than or equal to min_processing_units.
    minNodes: Minimum number of nodes allocated to the instance. If set, this
      number should be greater than or equal to 1.
    minProcessingUnits: Minimum number of processing units allocated to the
      instance. If set, this number should be multiples of 1000.
  """
    maxNodes = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    maxProcessingUnits = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    minNodes = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    minProcessingUnits = _messages.IntegerField(4, variant=_messages.Variant.INT32)