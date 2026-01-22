from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListConsentArtifactsResponse(_messages.Message):
    """A ListConsentArtifactsResponse object.

  Fields:
    consentArtifacts: The returned Consent artifacts. The maximum number of
      artifacts returned is determined by the value of page_size in the
      ListConsentArtifactsRequest.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    consentArtifacts = _messages.MessageField('ConsentArtifact', 1, repeated=True)
    nextPageToken = _messages.StringField(2)