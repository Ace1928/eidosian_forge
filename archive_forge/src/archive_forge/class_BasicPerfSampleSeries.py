from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BasicPerfSampleSeries(_messages.Message):
    """Encapsulates the metadata for basic sample series represented by a line
  chart

  Enums:
    PerfMetricTypeValueValuesEnum:
    PerfUnitValueValuesEnum:
    SampleSeriesLabelValueValuesEnum:

  Fields:
    perfMetricType: A PerfMetricTypeValueValuesEnum attribute.
    perfUnit: A PerfUnitValueValuesEnum attribute.
    sampleSeriesLabel: A SampleSeriesLabelValueValuesEnum attribute.
  """

    class PerfMetricTypeValueValuesEnum(_messages.Enum):
        """PerfMetricTypeValueValuesEnum enum type.

    Values:
      perfMetricTypeUnspecified: <no description>
      memory: <no description>
      cpu: <no description>
      network: <no description>
      graphics: <no description>
    """
        perfMetricTypeUnspecified = 0
        memory = 1
        cpu = 2
        network = 3
        graphics = 4

    class PerfUnitValueValuesEnum(_messages.Enum):
        """PerfUnitValueValuesEnum enum type.

    Values:
      perfUnitUnspecified: <no description>
      kibibyte: <no description>
      percent: <no description>
      bytesPerSecond: <no description>
      framesPerSecond: <no description>
      byte: <no description>
    """
        perfUnitUnspecified = 0
        kibibyte = 1
        percent = 2
        bytesPerSecond = 3
        framesPerSecond = 4
        byte = 5

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
    perfMetricType = _messages.EnumField('PerfMetricTypeValueValuesEnum', 1)
    perfUnit = _messages.EnumField('PerfUnitValueValuesEnum', 2)
    sampleSeriesLabel = _messages.EnumField('SampleSeriesLabelValueValuesEnum', 3)