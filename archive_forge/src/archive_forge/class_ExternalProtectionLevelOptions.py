from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalProtectionLevelOptions(_messages.Message):
    """ExternalProtectionLevelOptions stores a group of additional fields for
  configuring a CryptoKeyVersion that are specific to the EXTERNAL protection
  level and EXTERNAL_VPC protection levels.

  Fields:
    ekmConnectionKeyPath: The path to the external key material on the EKM
      when using EkmConnection e.g., "v0/my/key". Set this field instead of
      external_key_uri when using an EkmConnection.
    externalKeyUri: The URI for an external resource that this
      CryptoKeyVersion represents.
  """
    ekmConnectionKeyPath = _messages.StringField(1)
    externalKeyUri = _messages.StringField(2)