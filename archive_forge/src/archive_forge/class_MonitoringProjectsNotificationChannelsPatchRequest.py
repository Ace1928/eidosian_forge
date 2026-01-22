from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsPatchRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsPatchRequest object.

  Fields:
    name: The full REST resource name for this channel. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID] The
      [CHANNEL_ID] is automatically assigned by the server on creation.
    notificationChannel: A NotificationChannel resource to be passed as the
      request body.
    updateMask: The fields to update.
  """
    name = _messages.StringField(1, required=True)
    notificationChannel = _messages.MessageField('NotificationChannel', 2)
    updateMask = _messages.StringField(3)