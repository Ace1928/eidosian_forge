from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DeleteAction(_messages.Message):
    """Delete a file or directory.

  Fields:
    path: The path of the file or directory. If path refers to a directory,
      the directory and its contents are deleted.
  """
    path = _messages.StringField(1)