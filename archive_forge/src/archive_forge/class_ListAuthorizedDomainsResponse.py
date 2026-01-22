from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAuthorizedDomainsResponse(_messages.Message):
    """A list of Authorized Domains.

  Fields:
    domains: The authorized domains belonging to the user.
    nextPageToken: Continuation token for fetching the next page of results.
  """
    domains = _messages.MessageField('AuthorizedDomain', 1, repeated=True)
    nextPageToken = _messages.StringField(2)