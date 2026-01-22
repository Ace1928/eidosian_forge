from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileIODetails(_messages.Message):
    """Metadata for a File connector used by the job.

  Fields:
    filePattern: File Pattern used to access files by the connector.
  """
    filePattern = _messages.StringField(1)