from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SampleSeriesLabelValueValuesEnum(_messages.Enum):
    """SampleSeriesLabelValueValuesEnum enum type.

    Values:
      sampleSeriesTypeUnspecified: <no description>
      memoryRssPrivate: Memory sample series
      memoryRssShared: <no description>
      memoryRssTotal: <no description>
      memoryTotal: <no description>
      cpuUser: CPU sample series
      cpuKernel: <no description>
      cpuTotal: <no description>
      ntBytesTransferred: Network sample series
      ntBytesReceived: <no description>
      networkSent: <no description>
      networkReceived: <no description>
      graphicsFrameRate: Graphics sample series
    """
    sampleSeriesTypeUnspecified = 0
    memoryRssPrivate = 1
    memoryRssShared = 2
    memoryRssTotal = 3
    memoryTotal = 4
    cpuUser = 5
    cpuKernel = 6
    cpuTotal = 7
    ntBytesTransferred = 8
    ntBytesReceived = 9
    networkSent = 10
    networkReceived = 11
    graphicsFrameRate = 12