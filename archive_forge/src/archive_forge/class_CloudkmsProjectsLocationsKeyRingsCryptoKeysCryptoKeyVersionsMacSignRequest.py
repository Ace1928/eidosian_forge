from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacSignRequest(_messages.Message):
    """A
  CloudkmsProjectsLocationsKeyRingsCryptoKeysCryptoKeyVersionsMacSignRequest
  object.

  Fields:
    macSignRequest: A MacSignRequest resource to be passed as the request
      body.
    name: Required. The resource name of the CryptoKeyVersion to use for
      signing.
  """
    macSignRequest = _messages.MessageField('MacSignRequest', 1)
    name = _messages.StringField(2, required=True)