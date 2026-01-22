from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Folder(_messages.Message):
    """A folder. Only available in buckets with hierarchical namespace enabled.

  Messages:
    PendingRenameInfoValue: Only present if the folder is part of an ongoing
      rename folder operation. Contains information which can be used to query
      the operation status.

  Fields:
    bucket: The name of the bucket containing this folder.
    createTime: The creation time of the folder in RFC 3339 format.
    id: The ID of the folder, including the bucket name, folder name.
    kind: The kind of item this is. For folders, this is always
      storage#folder.
    metageneration: The version of the metadata for this folder. Used for
      preconditions and for detecting changes in metadata.
    name: The name of the folder. Required if not specified by URL parameter.
    pendingRenameInfo: Only present if the folder is part of an ongoing rename
      folder operation. Contains information which can be used to query the
      operation status.
    selfLink: The link to this folder.
    updateTime: The modification time of the folder metadata in RFC 3339
      format.
  """

    class PendingRenameInfoValue(_messages.Message):
        """Only present if the folder is part of an ongoing rename folder
    operation. Contains information which can be used to query the operation
    status.

    Fields:
      operationId: The ID of the rename folder operation.
    """
        operationId = _messages.StringField(1)
    bucket = _messages.StringField(1)
    createTime = _message_types.DateTimeField(2)
    id = _messages.StringField(3)
    kind = _messages.StringField(4, default='storage#folder')
    metageneration = _messages.IntegerField(5)
    name = _messages.StringField(6)
    pendingRenameInfo = _messages.MessageField('PendingRenameInfoValue', 7)
    selfLink = _messages.StringField(8)
    updateTime = _message_types.DateTimeField(9)