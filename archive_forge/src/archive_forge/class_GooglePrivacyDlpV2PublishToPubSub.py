from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PublishToPubSub(_messages.Message):
    """Publish a message into a given Pub/Sub topic when DlpJob has completed.
  The message contains a single field, `DlpJobName`, which is equal to the
  finished job's [`DlpJob.name`](https://cloud.google.com/sensitive-data-
  protection/docs/reference/rest/v2/projects.dlpJobs#DlpJob). Compatible with:
  Inspect, Risk

  Fields:
    topic: Cloud Pub/Sub topic to send notifications to. The topic must have
      given publishing access rights to the DLP API service account executing
      the long running DlpJob sending the notifications. Format is
      projects/{project}/topics/{topic}.
  """
    topic = _messages.StringField(1)