from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OpenTelemetryFormat(_messages.Message):
    """OpenTelemetryFormat contains metadata to help convert Cloud Trace trace
  format to OpenTelemetry trace format.

  Fields:
    version: OpenTelemetry format defined by https://github.com/open-
      telemetry/opentelemetry-proto/blob/main/opentelemetry/proto/collector/tr
      ace/v1/trace_service.proto Version of the OpenTelemetry schema as it
      appears in the defining proto package. Currently, only "v1" is
      supported, but support for more version may added in the future.
  """
    version = _messages.StringField(1)