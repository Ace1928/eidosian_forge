from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMembershipConstraintTemplatesResponse(_messages.Message):
    """Response schema for ListMembershipConstraintTemplates.

  Fields:
    membershipConstraintTemplates: List of membership-level constraint
      template info.
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    totalSize: The number of membership constraint templates in the response.
  """
    membershipConstraintTemplates = _messages.MessageField('MembershipConstraintTemplate', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3)