from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSecretsResponse(_messages.Message):
    """Response message for SecretManagerService.ListSecrets.

  Fields:
    nextPageToken: A token to retrieve the next page of results. Pass this
      value in ListSecretsRequest.page_token to retrieve the next page.
    secrets: The list of Secrets sorted in reverse by create_time (newest
      first).
    totalSize: The total number of Secrets but 0 when the
      ListSecretsRequest.filter field is set.
  """
    nextPageToken = _messages.StringField(1)
    secrets = _messages.MessageField('Secret', 2, repeated=True)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)