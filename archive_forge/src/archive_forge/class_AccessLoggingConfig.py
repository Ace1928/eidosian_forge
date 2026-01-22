from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessLoggingConfig(_messages.Message):
    """Access logging configuration.

  Fields:
    provider: Required. Must be a valid access logging provider specified in
      TelemetryProviders. Utmost 2 providers can be configured. Provider order
      will be respected.
    workloadContextSelector: Optional. Applies access logging configuration
      only to workloads selected by this workload context selector. If unset,
      applies to all workloads. Only for GSM.
  """
    provider = _messages.StringField(1, repeated=True)
    workloadContextSelector = _messages.MessageField('WorkloadContextSelector', 2)