from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KmsWrappedCryptoKey(_messages.Message):
    """Include to use an existing data crypto key wrapped by KMS. The wrapped
  key must be a 128-, 192-, or 256-bit key. The key must grant the Cloud IAM
  permission `cloudkms.cryptoKeyVersions.useToDecrypt` to the project's Cloud
  Healthcare Service Agent service account. For more information, see
  [Creating a wrapped key] (https://cloud.google.com/dlp/docs/create-wrapped-
  key).

  Fields:
    cryptoKey: Required. The resource name of the KMS CryptoKey to use for
      unwrapping. For example, `projects/{project_id}/locations/{location_id}/
      keyRings/{keyring}/cryptoKeys/{key}`.
    wrappedKey: Required. The wrapped data crypto key.
  """
    cryptoKey = _messages.StringField(1)
    wrappedKey = _messages.BytesField(2)