from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringProjectsNotificationChannelsCreateRequest(_messages.Message):
    """A MonitoringProjectsNotificationChannelsCreateRequest object.

  Fields:
    name: Required. The project
      (https://cloud.google.com/monitoring/api/v3#project_name) on which to
      execute the request. The format is: projects/[PROJECT_ID_OR_NUMBER] This
      names the container into which the channel will be written, this does
      not name the newly created channel. The resulting channel's name will
      have a normalized version of this field as a prefix, but will add
      /notificationChannels/[CHANNEL_ID] to identify the channel.
    notificationChannel: A NotificationChannel resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    notificationChannel = _messages.MessageField('NotificationChannel', 2)