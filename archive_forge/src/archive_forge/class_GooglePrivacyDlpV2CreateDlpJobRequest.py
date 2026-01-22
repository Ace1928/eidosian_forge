from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2CreateDlpJobRequest(_messages.Message):
    """Request message for CreateDlpJobRequest. Used to initiate long running
  jobs such as calculating risk metrics or inspecting Google Cloud Storage.

  Fields:
    inspectJob: An inspection job scans a storage repository for InfoTypes.
    jobId: The job id can contain uppercase and lowercase letters, numbers,
      and hyphens; that is, it must match the regular expression:
      `[a-zA-Z\\d-_]+`. The maximum length is 100 characters. Can be empty to
      allow the system to generate one.
    locationId: Deprecated. This field has no effect.
    riskJob: A risk analysis job calculates re-identification risk metrics for
      a BigQuery table.
  """
    inspectJob = _messages.MessageField('GooglePrivacyDlpV2InspectJobConfig', 1)
    jobId = _messages.StringField(2)
    locationId = _messages.StringField(3)
    riskJob = _messages.MessageField('GooglePrivacyDlpV2RiskAnalysisJobConfig', 4)