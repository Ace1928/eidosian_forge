from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerfEnvironment(_messages.Message):
    """Encapsulates performance environment info

  Fields:
    cpuInfo: CPU related environment info
    memoryInfo: Memory related environment info
  """
    cpuInfo = _messages.MessageField('CPUInfo', 1)
    memoryInfo = _messages.MessageField('MemoryInfo', 2)