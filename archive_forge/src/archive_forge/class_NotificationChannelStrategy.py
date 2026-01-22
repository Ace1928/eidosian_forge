from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationChannelStrategy(_messages.Message):
    """Control over how the notification channels in notification_channels are
  notified when this alert fires, on a per-channel basis.

  Fields:
    notificationChannelNames: The full REST resource name for the notification
      channels that these settings apply to. Each of these correspond to the
      name field in one of the NotificationChannel objects referenced in the
      notification_channels field of this AlertPolicy. The format is:
      projects/[PROJECT_ID_OR_NUMBER]/notificationChannels/[CHANNEL_ID]
    renotifyInterval: The frequency at which to send reminder notifications
      for open incidents.
  """
    notificationChannelNames = _messages.StringField(1, repeated=True)
    renotifyInterval = _messages.StringField(2)