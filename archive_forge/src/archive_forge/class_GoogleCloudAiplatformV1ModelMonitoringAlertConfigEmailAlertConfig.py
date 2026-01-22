from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1ModelMonitoringAlertConfigEmailAlertConfig(_messages.Message):
    """The config for email alert.

  Fields:
    userEmails: The email addresses to send the alert.
  """
    userEmails = _messages.StringField(1, repeated=True)