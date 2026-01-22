from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CopyAction(_messages.Message):
    """Copy the contents of a file or directory at from_path in the specified
  revision or snapshot to to_path.  To rename a file, copy it to the new path
  and delete the old.

  Fields:
    fromPath: The path to copy from.
    fromRevisionId: The revision ID from which to copy the file.
    fromSnapshotId: The snapshot ID from which to copy the file.
    toPath: The path to copy to.
  """
    fromPath = _messages.StringField(1)
    fromRevisionId = _messages.StringField(2)
    fromSnapshotId = _messages.StringField(3)
    toPath = _messages.StringField(4)