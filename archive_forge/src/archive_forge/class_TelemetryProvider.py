from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelemetryProvider(_messages.Message):
    """Telemetry provider configuration.

  Fields:
    cloudLogging: Optional. Specifies configuration to write access logs to
      Google Cloud Logging.
    cloudTracing: Optional. Specifies configuration to send traces to Google
      Cloud Tracing. Only for GSM.
    fileAccessLog: Optional. Specifies configuration to write access logs to
      the local filesystem.
    name: Required. A unique name identifying this telemetry provider.
  """
    cloudLogging = _messages.MessageField('TelemetryProviderCloudLogging', 1)
    cloudTracing = _messages.MessageField('TelemetryProviderCloudTracing', 2)
    fileAccessLog = _messages.MessageField('TelemetryProviderFileAccessLog', 3)
    name = _messages.StringField(4)