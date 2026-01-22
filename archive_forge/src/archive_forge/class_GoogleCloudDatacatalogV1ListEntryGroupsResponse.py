from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ListEntryGroupsResponse(_messages.Message):
    """Response message for ListEntryGroups.

  Fields:
    entryGroups: Entry group details.
    nextPageToken: Pagination token to specify in the next call to retrieve
      the next page of results. Empty if there are no more items.
  """
    entryGroups = _messages.MessageField('GoogleCloudDatacatalogV1EntryGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)