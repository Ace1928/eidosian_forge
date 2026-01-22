from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsRemotebuildbotResourceUsage(_messages.Message):
    """ResourceUsage is the system resource usage of the host machine.

  Enums:
    BotStateValueValuesEnum:

  Fields:
    botState: A BotStateValueValuesEnum attribute.
    cpuUsedPercent: A number attribute.
    diskUsage: A GoogleDevtoolsRemotebuildbotResourceUsageStat attribute.
    dockerRootDiskUsage: A GoogleDevtoolsRemotebuildbotResourceUsageStat
      attribute.
    memoryUsage: A GoogleDevtoolsRemotebuildbotResourceUsageStat attribute.
    totalDiskIoStats: A GoogleDevtoolsRemotebuildbotResourceUsageIOStats
      attribute.
  """

    class BotStateValueValuesEnum(_messages.Enum):
        """BotStateValueValuesEnum enum type.

    Values:
      UNSPECIFIED: <no description>
      IDLE: <no description>
      BUSY: <no description>
    """
        UNSPECIFIED = 0
        IDLE = 1
        BUSY = 2
    botState = _messages.EnumField('BotStateValueValuesEnum', 1)
    cpuUsedPercent = _messages.FloatField(2)
    diskUsage = _messages.MessageField('GoogleDevtoolsRemotebuildbotResourceUsageStat', 3)
    dockerRootDiskUsage = _messages.MessageField('GoogleDevtoolsRemotebuildbotResourceUsageStat', 4)
    memoryUsage = _messages.MessageField('GoogleDevtoolsRemotebuildbotResourceUsageStat', 5)
    totalDiskIoStats = _messages.MessageField('GoogleDevtoolsRemotebuildbotResourceUsageIOStats', 6)