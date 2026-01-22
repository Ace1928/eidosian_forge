from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListGuestPoliciesResponse(_messages.Message):
    """A response message for listing guest policies.

  Fields:
    guestPolicies: The list of GuestPolicies.
    nextPageToken: A pagination token that can be used to get the next page of
      guest policies.
  """
    guestPolicies = _messages.MessageField('GuestPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)