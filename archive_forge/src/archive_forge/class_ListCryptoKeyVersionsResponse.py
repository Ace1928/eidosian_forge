from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCryptoKeyVersionsResponse(_messages.Message):
    """Response message for KeyManagementService.ListCryptoKeyVersions.

  Fields:
    cryptoKeyVersions: The list of CryptoKeyVersions.
    nextPageToken: A token to retrieve next page of results. Pass this value
      in ListCryptoKeyVersionsRequest.page_token to retrieve the next page of
      results.
    totalSize: The total number of CryptoKeyVersions that matched the query.
  """
    cryptoKeyVersions = _messages.MessageField('CryptoKeyVersion', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)