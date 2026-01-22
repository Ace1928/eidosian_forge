from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TextFormat(_messages.Message):
    """Configuration for reading Cloud Storage data in text format. Each line
  of text as specified by the delimiter will be set to the `data` field of a
  Pub/Sub message.

  Fields:
    delimiter: Optional. When unset, '\\n' is used.
  """
    delimiter = _messages.StringField(1)