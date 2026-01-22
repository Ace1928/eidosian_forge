from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageFoldersGetRequest(_messages.Message):
    """A StorageFoldersGetRequest object.

  Fields:
    bucket: Name of the bucket in which the folder resides.
    folder: Name of a folder.
    ifMetagenerationMatch: Makes the return of the folder metadata conditional
      on whether the folder's current metageneration matches the given value.
    ifMetagenerationNotMatch: Makes the return of the folder metadata
      conditional on whether the folder's current metageneration does not
      match the given value.
  """
    bucket = _messages.StringField(1, required=True)
    folder = _messages.StringField(2, required=True)
    ifMetagenerationMatch = _messages.IntegerField(3)
    ifMetagenerationNotMatch = _messages.IntegerField(4)