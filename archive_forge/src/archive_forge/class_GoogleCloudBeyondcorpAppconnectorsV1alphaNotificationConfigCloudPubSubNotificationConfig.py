from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudBeyondcorpAppconnectorsV1alphaNotificationConfigCloudPubSubNotificationConfig(_messages.Message):
    """The configuration for Pub/Sub messaging for the AppConnector.

  Fields:
    pubsubSubscription: The Pub/Sub subscription the AppConnector uses to
      receive notifications.
  """
    pubsubSubscription = _messages.StringField(1)