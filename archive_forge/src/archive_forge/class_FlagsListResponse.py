from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FlagsListResponse(_messages.Message):
    """Flags list response.

  Fields:
    items: List of flags.
    kind: This is always `sql#flagsList`.
  """
    items = _messages.MessageField('Flag', 1, repeated=True)
    kind = _messages.StringField(2)