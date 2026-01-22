from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListNoteOccurrencesResponse(_messages.Message):
    """Response for listing occurrences for a note.

  Fields:
    nextPageToken: Token to provide to skip to a particular spot in the list.
    occurrences: The occurrences attached to the specified note.
  """
    nextPageToken = _messages.StringField(1)
    occurrences = _messages.MessageField('Occurrence', 2, repeated=True)