from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListServiceAccountKeysResponse(_messages.Message):
    """The service account keys list response.

  Fields:
    keys: The public keys for the service account.
  """
    keys = _messages.MessageField('ServiceAccountKey', 1, repeated=True)