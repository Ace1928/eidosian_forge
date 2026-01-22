from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomerManagedEncryption(_messages.Message):
    """Configuration for encrypting secret payloads using customer-managed
  encryption keys (CMEK).

  Fields:
    kmsKeyName: Required. The resource name of the Cloud KMS CryptoKey used to
      encrypt secret payloads. For secrets using the UserManaged replication
      policy type, Cloud KMS CryptoKeys must reside in the same location as
      the replica location. For secrets using the Automatic replication policy
      type, Cloud KMS CryptoKeys must reside in `global`. The expected format
      is `projects/*/locations/*/keyRings/*/cryptoKeys/*`.
  """
    kmsKeyName = _messages.StringField(1)