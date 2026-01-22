from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRestorePlansResponse(_messages.Message):
    """Response message for ListRestorePlans.

  Fields:
    nextPageToken: A token which may be sent as page_token in a subsequent
      `ListRestorePlans` call to retrieve the next page of results. If this
      field is omitted or empty, then there are no more results to return.
    restorePlans: The list of RestorePlans matching the given criteria.
    unreachable: Locations that could not be reached.
  """
    nextPageToken = _messages.StringField(1)
    restorePlans = _messages.MessageField('RestorePlan', 2, repeated=True)
    unreachable = _messages.StringField(3, repeated=True)