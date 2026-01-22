from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TracingConfig(_messages.Message):
    """Tracing configuration. Only for GSM.

  Fields:
    provider: Required. Must be a valid tracing provider specified in
      TelemetryProviders. Only for GSM.
    workloadContextSelector: Optional. Applies tracing configuration only to
      workloads selected by this workload context selector. If unset, applies
      to all workloads. Only for GSM.
  """
    provider = _messages.StringField(1)
    workloadContextSelector = _messages.MessageField('WorkloadContextSelector', 2)