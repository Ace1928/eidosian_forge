from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest(_messages.Message):
    """A CloudkmsProjectsLocationsKeyRingsCryptoKeysCreateRequest object.

  Fields:
    cryptoKey: A CryptoKey resource to be passed as the request body.
    cryptoKeyId: Required. It must be unique within a KeyRing and match the
      regular expression `[a-zA-Z0-9_-]{1,63}`
    parent: Required. The name of the KeyRing associated with the CryptoKeys.
  """
    cryptoKey = _messages.MessageField('CryptoKey', 1)
    cryptoKeyId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)