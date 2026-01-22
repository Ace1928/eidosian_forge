from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TiersListResponse(_messages.Message):
    """Tiers list response.

  Fields:
    items: List of tiers.
    kind: This is always `sql#tiersList`.
  """
    items = _messages.MessageField('Tier', 1, repeated=True)
    kind = _messages.StringField(2)