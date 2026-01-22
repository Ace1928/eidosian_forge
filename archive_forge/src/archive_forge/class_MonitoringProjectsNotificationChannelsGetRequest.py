from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsGetRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsGetRequest object.

  Fields:
    name: Required. The channel for which to execute the request. The format
      is: projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
  """
    name = _messages.StringField(1, required=True)