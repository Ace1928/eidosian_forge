from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConsentRevisionsResponse(_messages.Message):
    """A ListConsentRevisionsResponse object.

  Fields:
    consents: The returned Consent revisions. The maximum number of revisions
      returned is determined by the value of `page_size` in the
      ListConsentRevisionsRequest.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    consents = _messages.MessageField('Consent', 1, repeated=True)
    nextPageToken = _messages.StringField(2)