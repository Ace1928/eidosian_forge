from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest(_messages.Message):
    """A
  CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsCreateRequest
  object.

  Fields:
    cryptoKeyVersion: A CryptoKeyVersion resource to be passed as the request
      body.
    parent: Required. The name of the CryptoKey associated with the
      CryptoKeyVersions.
  """
    cryptoKeyVersion = _messages.MessageField('CryptoKeyVersion', 1)
    parent = _messages.StringField(2, required=True)