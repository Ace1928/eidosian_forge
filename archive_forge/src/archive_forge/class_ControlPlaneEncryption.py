from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ControlPlaneEncryption(_messages.Message):
    """Configuration for Customer-managed KMS key support for remote control
  plane cluster disk encryption.

  Enums:
    KmsKeyStateValueValuesEnum: Output only. Availability of the Cloud KMS
      CryptoKey. If not `KEY_AVAILABLE`, then nodes may go offline as they
      cannot access their local data. This can be caused by a lack of
      permissions to use the key, or if the key is disabled or deleted.

  Fields:
    kmsKey: Immutable. The Cloud KMS CryptoKey e.g. projects/{project}/locatio
      ns/{location}/keyRings/{keyRing}/cryptoKeys/{cryptoKey} to use for
      protecting control plane disks. If not specified, a Google-managed key
      will be used instead.
    kmsKeyActiveVersion: Output only. The Cloud KMS CryptoKeyVersion currently
      in use for protecting control plane disks. Only applicable if kms_key is
      set.
    kmsKeyState: Output only. Availability of the Cloud KMS CryptoKey. If not
      `KEY_AVAILABLE`, then nodes may go offline as they cannot access their
      local data. This can be caused by a lack of permissions to use the key,
      or if the key is disabled or deleted.
    kmsStatus: Output only. Error status returned by Cloud KMS when using this
      key. This field may be populated only if `kms_key_state` is not
      `KMS_KEY_STATE_KEY_AVAILABLE`. If populated, this field contains the
      error status reported by Cloud KMS.
  """

    class KmsKeyStateValueValuesEnum(_messages.Enum):
        """Output only. Availability of the Cloud KMS CryptoKey. If not
    `KEY_AVAILABLE`, then nodes may go offline as they cannot access their
    local data. This can be caused by a lack of permissions to use the key, or
    if the key is disabled or deleted.

    Values:
      KMS_KEY_STATE_UNSPECIFIED: Unspecified.
      KMS_KEY_STATE_KEY_AVAILABLE: The key is available for use, and dependent
        resources should be accessible.
      KMS_KEY_STATE_KEY_UNAVAILABLE: The key is unavailable for an unspecified
        reason. Dependent resources may be inaccessible.
    """
        KMS_KEY_STATE_UNSPECIFIED = 0
        KMS_KEY_STATE_KEY_AVAILABLE = 1
        KMS_KEY_STATE_KEY_UNAVAILABLE = 2
    kmsKey = _messages.StringField(1)
    kmsKeyActiveVersion = _messages.StringField(2)
    kmsKeyState = _messages.EnumField('KmsKeyStateValueValuesEnum', 3)
    kmsStatus = _messages.MessageField('Status', 4)