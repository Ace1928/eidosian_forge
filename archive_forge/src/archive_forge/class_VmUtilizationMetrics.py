from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmUtilizationMetrics(_messages.Message):
    """Utilization metrics values for a single VM.

  Fields:
    cpuAveragePercent: Average CPU usage, percent.
    cpuMaxPercent: Max CPU usage, percent.
    diskIoRateAverageKbps: Average disk IO rate, in kilobytes per second.
    diskIoRateMaxKbps: Max disk IO rate, in kilobytes per second.
    memoryAveragePercent: Average memory usage, percent.
    memoryMaxPercent: Max memory usage, percent.
    networkThroughputAverageKbps: Average network throughput (combined
      transmit-rates and receive-rates), in kilobytes per second.
    networkThroughputMaxKbps: Max network throughput (combined transmit-rates
      and receive-rates), in kilobytes per second.
  """
    cpuAveragePercent = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    cpuMaxPercent = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    diskIoRateAverageKbps = _messages.IntegerField(3)
    diskIoRateMaxKbps = _messages.IntegerField(4)
    memoryAveragePercent = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    memoryMaxPercent = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    networkThroughputAverageKbps = _messages.IntegerField(7)
    networkThroughputMaxKbps = _messages.IntegerField(8)