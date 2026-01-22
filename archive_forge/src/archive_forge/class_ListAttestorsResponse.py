from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListAttestorsResponse(_messages.Message):
    """Response message for BinauthzManagementService.ListAttestors.

  Fields:
    attestors: The list of attestors.
    nextPageToken: A token to retrieve the next page of results. Pass this
      value in the ListAttestorsRequest.page_token field in the subsequent
      call to the `ListAttestors` method to retrieve the next page of results.
  """
    attestors = _messages.MessageField('Attestor', 1, repeated=True)
    nextPageToken = _messages.StringField(2)