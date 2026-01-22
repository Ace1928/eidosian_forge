from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysEncryptRequest object.

  Fields:
    encryptRequest: A EncryptRequest resource to be passed as the request
      body.
    name: Required. The resource name of the CryptoKey or CryptoKeyVersion to
      use for encryption.  If a CryptoKey is specified, the server will use
      its primary version.
  """
    encryptRequest = _messages.MessageField('EncryptRequest', 1)
    name = _messages.StringField(2, required=True)