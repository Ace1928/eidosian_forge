from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class CoordinatesValueListEntry(_messages.Message):
    """Single entry in a CoordinatesValue.

    Messages:
      EntryValueListEntry: Single entry in a EntryValue.

    Fields:
      entry: A EntryValueListEntry attribute.
    """

    class EntryValueListEntry(_messages.Message):
        """Single entry in a EntryValue.

      Fields:
        entry: A number attribute.
      """
        entry = _messages.FloatField(1, repeated=True)
    entry = _messages.MessageField('EntryValueListEntry', 1, repeated=True)