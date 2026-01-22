from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacVerifyRequest(_messages.Message):
    """A
  CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacVerifyRequest
  object.

  Fields:
    macVerifyRequest: A MacVerifyRequest resource to be passed as the request
      body.
    name: Required. The resource name of the CryptoKeyVersion to use for
      verification.
  """
    macVerifyRequest = _messages.MessageField('MacVerifyRequest', 1)
    name = _messages.StringField(2, required=True)