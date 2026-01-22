from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageFoldersRenameRequest(_messages.Message):
    """A StorageFoldersRenameRequest object.

  Fields:
    bucket: Name of the bucket in which the folders are in.
    destinationFolder: Name of the destination folder.
    ifSourceMetagenerationMatch: Makes the operation conditional on whether
      the source object's current metageneration matches the given value.
    ifSourceMetagenerationNotMatch: Makes the operation conditional on whether
      the source object's current metageneration does not match the given
      value.
    sourceFolder: Name of the source folder.
  """
    bucket = _messages.StringField(1, required=True)
    destinationFolder = _messages.StringField(2, required=True)
    ifSourceMetagenerationMatch = _messages.IntegerField(3)
    ifSourceMetagenerationNotMatch = _messages.IntegerField(4)
    sourceFolder = _messages.StringField(5, required=True)