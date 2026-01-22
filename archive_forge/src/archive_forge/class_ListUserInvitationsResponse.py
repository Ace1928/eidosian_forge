from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListUserInvitationsResponse(_messages.Message):
    """Response message for UserInvitation listing request.

  Fields:
    nextPageToken: The token for the next page. If not empty, indicates that
      there may be more `UserInvitation` resources that match the listing
      request; this value can be used in a subsequent
      ListUserInvitationsRequest to get continued results with the current
      list call.
    userInvitations: The list of UserInvitation resources.
  """
    nextPageToken = _messages.StringField(1)
    userInvitations = _messages.MessageField('UserInvitation', 2, repeated=True)