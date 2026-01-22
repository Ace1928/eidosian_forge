from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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