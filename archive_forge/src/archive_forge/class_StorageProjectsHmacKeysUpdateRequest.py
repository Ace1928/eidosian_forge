from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageProjectsHmacKeysUpdateRequest(_messages.Message):
    """A StorageProjectsHmacKeysUpdateRequest object.

  Fields:
    projectId: Project ID
    accessId: Name of the HMAC key being updated.
  """
    projectId = _messages.StringField(1, required=True)
    accessId = _messages.StringField(2, required=True)
    resource = _messages.MessageField('HmacKeyMetadata', 3)