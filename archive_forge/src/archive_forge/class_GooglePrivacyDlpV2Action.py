from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Action(_messages.Message):
    """A task to execute on the completion of a job. See
  https://cloud.google.com/sensitive-data-protection/docs/concepts-actions to
  learn more.

  Fields:
    deidentify: Create a de-identified copy of the input data.
    jobNotificationEmails: Sends an email when the job completes. The email
      goes to IAM project owners and technical [Essential
      Contacts](https://cloud.google.com/resource-manager/docs/managing-
      notification-contacts).
    pubSub: Publish a notification to a Pub/Sub topic.
    publishFindingsToCloudDataCatalog: Publish findings to Cloud Datahub.
    publishSummaryToCscc: Publish summary to Cloud Security Command Center
      (Alpha).
    publishToStackdriver: Enable Stackdriver metric
      dlp.googleapis.com/finding_count.
    saveFindings: Save resulting findings in a provided location.
  """
    deidentify = _messages.MessageField('GooglePrivacyDlpV2Deidentify', 1)
    jobNotificationEmails = _messages.MessageField('GooglePrivacyDlpV2JobNotificationEmails', 2)
    pubSub = _messages.MessageField('GooglePrivacyDlpV2PublishToPubSub', 3)
    publishFindingsToCloudDataCatalog = _messages.MessageField('GooglePrivacyDlpV2PublishFindingsToCloudDataCatalog', 4)
    publishSummaryToCscc = _messages.MessageField('GooglePrivacyDlpV2PublishSummaryToCscc', 5)
    publishToStackdriver = _messages.MessageField('GooglePrivacyDlpV2PublishToStackdriver', 6)
    saveFindings = _messages.MessageField('GooglePrivacyDlpV2SaveFindings', 7)