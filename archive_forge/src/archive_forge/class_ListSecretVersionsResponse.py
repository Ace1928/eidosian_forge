from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSecretVersionsResponse(_messages.Message):
    """Response message for SecretManagerService.ListSecretVersions.

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass this
      value in ListSecretVersionsRequest.page_token to retrieve the next page.
    totalSize: The total number of SecretVersions but 0 when the
      ListSecretsRequest.filter field is set.
    versions: The list of SecretVersions sorted in reverse by create_time
      (newest first).
  """
    nextPageToken = _messages.StringField(1)
    totalSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    versions = _messages.MessageField('SecretVersion', 3, repeated=True)