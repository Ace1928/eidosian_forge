from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CryptoKeyVersion(_messages.Message):
    """A CryptoKeyVersion represents an individual cryptographic key, and the
  associated key material.  It can be used for cryptographic operations either
  directly, or via its parent CryptoKey, in which case the server will choose
  the appropriate version for the operation.  For security reasons, the raw
  cryptographic key material represented by a CryptoKeyVersion can never be
  viewed or exported. It can only be used to encrypt or decrypt data when an
  authorized user or application invokes Cloud KMS.

  Enums:
    StateValueValuesEnum: The current state of the CryptoKeyVersion.

  Fields:
    createTime: Output only. The time at which this CryptoKeyVersion was
      created.
    destroyEventTime: Output only. The time this CryptoKeyVersion's key
      material was destroyed. Only present if state is DESTROYED.
    destroyTime: Output only. The time this CryptoKeyVersion's key material is
      scheduled for destruction. Only present if state is DESTROY_SCHEDULED.
    name: Output only. The resource name for this CryptoKeyVersion in the
      format
      `projects/*/locations/*/keyRings/*/cryptoKeys/*/cryptoKeyVersions/*`.
    state: The current state of the CryptoKeyVersion.
  """

    class StateValueValuesEnum(_messages.Enum):
        """The current state of the CryptoKeyVersion.

    Values:
      CRYPTO_KEY_VERSION_STATE_UNSPECIFIED: Not specified.
      ENABLED: This version may be used in Encrypt and Decrypt requests.
      DISABLED: This version may not be used, but the key material is still
        available, and the version can be placed back into the ENABLED state.
      DESTROYED: This version is destroyed, and the key material is no longer
        stored. A version may not leave this state once entered.
      DESTROY_SCHEDULED: This version is scheduled for destruction, and will
        be destroyed soon. Call RestoreCryptoKeyVersion to put it back into
        the DISABLED state.
    """
        CRYPTO_KEY_VERSION_STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        DESTROYED = 3
        DESTROY_SCHEDULED = 4
    createTime = _messages.StringField(1)
    destroyEventTime = _messages.StringField(2)
    destroyTime = _messages.StringField(3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)