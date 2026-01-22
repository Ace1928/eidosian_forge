from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TelemetryProviderCloudTracing(_messages.Message):
    """Google Cloud Tracing provider configuration. Default (empty)
  configuration sends traces to Google Cloud Tracing. Only for GSM.
  """