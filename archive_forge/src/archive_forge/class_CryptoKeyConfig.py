from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CryptoKeyConfig(_messages.Message):
    """The crypto key configuration. This field is used by the Customer-managed
  encryption keys (CMEK) feature.

  Fields:
    keyReference: The name of the key which is used to encrypt/decrypt
      customer data. For key in Cloud KMS, the key should be in the format of
      `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
  """
    keyReference = _messages.StringField(1)