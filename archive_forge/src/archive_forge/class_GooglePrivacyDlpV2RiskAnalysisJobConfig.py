from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2RiskAnalysisJobConfig(_messages.Message):
    """Configuration for a risk analysis job. See
  https://cloud.google.com/sensitive-data-protection/docs/concepts-risk-
  analysis to learn more.

  Fields:
    actions: Actions to execute at the completion of the job. Are executed in
      the order provided.
    privacyMetric: Privacy metric to compute.
    sourceTable: Input dataset to compute metrics over.
  """
    actions = _messages.MessageField('GooglePrivacyDlpV2Action', 1, repeated=True)
    privacyMetric = _messages.MessageField('GooglePrivacyDlpV2PrivacyMetric', 2)
    sourceTable = _messages.MessageField('GooglePrivacyDlpV2BigQueryTable', 3)