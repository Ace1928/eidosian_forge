from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchGroupsResponse(_messages.Message):
    """The response message for GroupsService.SearchGroups.

  Fields:
    groups: The `Group` resources that match the search query.
    nextPageToken: A continuation token to retrieve the next page of results,
      or empty if there are no more results available.
  """
    groups = _messages.MessageField('Group', 1, repeated=True)
    nextPageToken = _messages.StringField(2)