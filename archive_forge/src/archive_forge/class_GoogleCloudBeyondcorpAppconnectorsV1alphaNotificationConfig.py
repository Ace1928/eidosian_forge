from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1alphaNotificationConfig(_messages.Message):
    """NotificationConfig defines the mechanisms to notify instance agent.

  Fields:
    pubsubNotification: Cloud Pub/Sub Configuration to receive notifications.
  """
    pubsubNotification = _messages.MessageField('GoogleCloudBeyondcorpAppconnectorsV1alphaNotificationConfigCloudPubSubNotificationConfig', 1)