from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1TraceConfig(_messages.Message):
    """TraceConfig defines the configurations in an environment of distributed
  trace.

  Enums:
    ExporterValueValuesEnum: Required. Exporter that is used to view the
      distributed trace captured using OpenCensus. An exporter sends traces to
      any backend that is capable of consuming them. Recorded spans can be
      exported by registered exporters.

  Fields:
    endpoint: Required. Endpoint of the exporter.
    exporter: Required. Exporter that is used to view the distributed trace
      captured using OpenCensus. An exporter sends traces to any backend that
      is capable of consuming them. Recorded spans can be exported by
      registered exporters.
    samplingConfig: Distributed trace configuration for all API proxies in an
      environment. You can also override the configuration for a specific API
      proxy using the distributed trace configuration overrides API.
  """

    class ExporterValueValuesEnum(_messages.Enum):
        """Required. Exporter that is used to view the distributed trace captured
    using OpenCensus. An exporter sends traces to any backend that is capable
    of consuming them. Recorded spans can be exported by registered exporters.

    Values:
      EXPORTER_UNSPECIFIED: Exporter unspecified
      JAEGER: Jaeger exporter
      CLOUD_TRACE: Cloudtrace exporter
    """
        EXPORTER_UNSPECIFIED = 0
        JAEGER = 1
        CLOUD_TRACE = 2
    endpoint = _messages.StringField(1)
    exporter = _messages.EnumField('ExporterValueValuesEnum', 2)
    samplingConfig = _messages.MessageField('GoogleCloudApigeeV1TraceSamplingConfig', 3)