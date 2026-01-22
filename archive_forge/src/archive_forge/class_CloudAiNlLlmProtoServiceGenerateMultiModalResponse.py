from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceGenerateMultiModalResponse(_messages.Message):
    """A CloudAiNlLlmProtoServiceGenerateMultiModalResponse object.

  Fields:
    candidates: Possible candidate responses to the conversation up until this
      point.
    facts: External facts retrieved for factuality/grounding.
    promptFeedback: Content filter results for a prompt sent in the request.
      Note: Sent only in the first stream chunk. Only happens when no
      candidates were generated due to content violations.
    reportingMetrics: Billable prediction metrics.
    usageMetadata: Usage metadata about the response(s).
  """
    candidates = _messages.MessageField('CloudAiNlLlmProtoServiceCandidate', 1, repeated=True)
    facts = _messages.MessageField('CloudAiNlLlmProtoServiceFact', 2, repeated=True)
    promptFeedback = _messages.MessageField('CloudAiNlLlmProtoServicePromptFeedback', 3)
    reportingMetrics = _messages.MessageField('IntelligenceCloudAutomlXpsReportingMetrics', 4)
    usageMetadata = _messages.MessageField('CloudAiNlLlmProtoServiceUsageMetadata', 5)