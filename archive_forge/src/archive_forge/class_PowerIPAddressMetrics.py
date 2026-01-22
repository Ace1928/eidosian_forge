from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PowerIPAddressMetrics(_messages.Message):
    """Power IP Address Metrics

  Fields:
    available: Number of available IP address
    total: Size of IP address space
    used: Number of used IP addresses
    utilization: Utilization for IP address
  """
    available = _messages.IntegerField(1)
    total = _messages.IntegerField(2)
    used = _messages.IntegerField(3)
    utilization = _messages.IntegerField(4)