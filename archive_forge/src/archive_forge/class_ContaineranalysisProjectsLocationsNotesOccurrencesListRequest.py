from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsLocationsNotesOccurrencesListRequest(_messages.Message):
    """A ContaineranalysisProjectsLocationsNotesOccurrencesListRequest object.

  Fields:
    filter: The filter expression.
    name: Required. The name of the note to list occurrences for in the form
      of `projects/[PROVIDER_ID]/notes/[NOTE_ID]`.
    pageSize: Number of occurrences to return in the list.
    pageToken: Token to provide to skip to a particular spot in the list.
  """
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)