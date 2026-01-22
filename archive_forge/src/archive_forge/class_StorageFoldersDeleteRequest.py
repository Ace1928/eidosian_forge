from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageFoldersDeleteRequest(_messages.Message):
    """A StorageFoldersDeleteRequest object.

  Fields:
    bucket: Name of the bucket in which the folder resides.
    folder: Name of a folder.
    ifMetagenerationMatch: If set, only deletes the folder if its
      metageneration matches this value.
    ifMetagenerationNotMatch: If set, only deletes the folder if its
      metageneration does not match this value.
  """
    bucket = _messages.StringField(1, required=True)
    folder = _messages.StringField(2, required=True)
    ifMetagenerationMatch = _messages.IntegerField(3)
    ifMetagenerationNotMatch = _messages.IntegerField(4)