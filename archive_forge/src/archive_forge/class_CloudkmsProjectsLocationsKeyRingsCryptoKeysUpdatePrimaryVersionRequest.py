from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysUpdatePrimaryVersionRequest
  object.

  Fields:
    name: The resource name of the CryptoKey to update.
    updateCryptoKeyPrimaryVersionRequest: A
      UpdateCryptoKeyPrimaryVersionRequest resource to be passed as the
      request body.
  """
    name = _messages.StringField(1, required=True)
    updateCryptoKeyPrimaryVersionRequest = _messages.MessageField('UpdateCryptoKeyPrimaryVersionRequest', 2)