from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2CloudLoggingEntry(_messages.Message):
    """Metadata taken from a [Cloud Logging LogEntry](https://cloud.google.com/
  logging/docs/reference/v2/rest/v2/LogEntry)

  Fields:
    insertId: A unique identifier for the log entry.
    logId: The type of the log (part of `log_name`. `log_name` is the resource
      name of the log to which this log entry belongs). For example:
      `cloudresourcemanager.googleapis.com/activity` Note that this field is
      not URL-encoded, unlike in `LogEntry`.
    resourceContainer: The organization, folder, or project of the monitored
      resource that produced this log entry.
    timestamp: The time the event described by the log entry occurred.
  """
    insertId = _messages.StringField(1)
    logId = _messages.StringField(2)
    resourceContainer = _messages.StringField(3)
    timestamp = _messages.StringField(4)