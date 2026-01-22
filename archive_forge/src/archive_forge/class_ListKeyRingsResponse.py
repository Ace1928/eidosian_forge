from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListKeyRingsResponse(_messages.Message):
    """Response message for KeyManagementService.ListKeyRings.

  Fields:
    keyRings: The list of KeyRings.
    nextPageToken: A token to retrieve next page of results. Pass this value
      in ListKeyRingsRequest.page_token to retrieve the next page of results.
    totalSize: The total number of KeyRings that matched the query.
  """
    keyRings = _messages.MessageField('KeyRing', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)