from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DirectoryEntry(_messages.Message):
    """Information about a directory.

  Fields:
    info: Information about the entry.
    isDir: Whether the entry is a file or directory.
    lastModifiedRevisionId: ID of the revision that most recently modified
      this file.
    name: Name of the entry relative to the directory.
  """
    info = _messages.MessageField('FileInfo', 1)
    isDir = _messages.BooleanField(2)
    lastModifiedRevisionId = _messages.StringField(3)
    name = _messages.StringField(4)