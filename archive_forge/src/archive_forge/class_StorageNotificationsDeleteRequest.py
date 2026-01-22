from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageNotificationsDeleteRequest(_messages.Message):
    """A StorageNotificationsDeleteRequest object.

  Fields:
    bucket: The parent bucket of the notification.
    notification: ID of the notification to delete.
    userProject: The project to be billed for this request. Required for
      Requester Pays buckets.
  """
    bucket = _messages.StringField(1, required=True)
    notification = _messages.StringField(2, required=True)
    userProject = _messages.StringField(3)