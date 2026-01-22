from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAuthorizationPoliciesResponse(_messages.Message):
    """Response returned by the ListAuthorizationPolicies method.

  Fields:
    authorizationPolicies: List of AuthorizationPolicies resources.
    nextPageToken: If there might be more results than those appearing in this
      response, then `next_page_token` is included. To get the next set of
      results, call this method again using the value of `next_page_token` as
      `page_token`.
  """
    authorizationPolicies = _messages.MessageField('AuthorizationPolicy', 1, repeated=True)
    nextPageToken = _messages.StringField(2)