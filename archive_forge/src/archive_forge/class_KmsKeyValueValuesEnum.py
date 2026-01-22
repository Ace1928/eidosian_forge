from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KmsKeyValueValuesEnum(_messages.Enum):
    """Specifies how each object's Cloud KMS customer-managed encryption key
    (CMEK) is preserved for transfers between Google Cloud Storage buckets. If
    unspecified, the default behavior is the same as
    KMS_KEY_DESTINATION_BUCKET_DEFAULT.

    Values:
      KMS_KEY_UNSPECIFIED: KmsKey behavior is unspecified.
      KMS_KEY_DESTINATION_BUCKET_DEFAULT: Use the destination bucket's default
        encryption settings.
      KMS_KEY_PRESERVE: Preserve the object's original Cloud KMS customer-
        managed encryption key (CMEK) if present. Objects that do not use a
        Cloud KMS encryption key will be encrypted using the destination
        bucket's encryption settings.
    """
    KMS_KEY_UNSPECIFIED = 0
    KMS_KEY_DESTINATION_BUCKET_DEFAULT = 1
    KMS_KEY_PRESERVE = 2