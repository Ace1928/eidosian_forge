from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListEntriesResponse(_messages.Message):
    """A GoogleCloudDataplexV1ListEntriesResponse object.

  Fields:
    entries: The list of entries.
    nextPageToken: Pagination token.
  """
    entries = _messages.MessageField('GoogleCloudDataplexV1Entry', 1, repeated=True)
    nextPageToken = _messages.StringField(2)