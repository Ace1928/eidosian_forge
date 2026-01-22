from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMulticastGroupsResponse(_messages.Message):
    """Response message for ListMulticastGroups.

  Fields:
    multicastGroups: The list of multicast groups.
    nextPageToken: A page token from an earlier query, as returned in
      `next_page_token`.
    unreachable: Locations that could not be reached.
  """
    multicastGroups = _messages.MessageField('MulticastGroup', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)