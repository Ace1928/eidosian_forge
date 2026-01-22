from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateBackupEncryptionConfig(_messages.Message):
    """Encryption configuration for the backup to create.

  Enums:
    EncryptionTypeValueValuesEnum: Required. The encryption type of the
      backup.

  Fields:
    encryptionType: Required. The encryption type of the backup.
    kmsKeyName: Optional. The Cloud KMS key that will be used to protect the
      backup. This field should be set only when encryption_type is
      `CUSTOMER_MANAGED_ENCRYPTION`. Values are of the form
      `projects//locations//keyRings//cryptoKeys/`.
    kmsKeyNames: Optional. Specifies the KMS configuration for the one or more
      keys used to protect the backup. Values are of the form
      `projects//locations//keyRings//cryptoKeys/`. The keys referenced by
      kms_key_names must fully cover all regions of the backup's instance
      configuration. Some examples: * For single region instance configs,
      specify a single regional location KMS key. * For multi-regional
      instance configs of type GOOGLE_MANAGED, either specify a multi-regional
      location KMS key or multiple regional location KMS keys that cover all
      regions in the instance config. * For an instance config of type
      USER_MANAGED, please specify only regional location KMS keys to cover
      each region in the instance config. Multi-regional location KMS keys are
      not supported for USER_MANAGED instance configs.
  """

    class EncryptionTypeValueValuesEnum(_messages.Enum):
        """Required. The encryption type of the backup.

    Values:
      ENCRYPTION_TYPE_UNSPECIFIED: Unspecified. Do not use.
      USE_DATABASE_ENCRYPTION: Use the same encryption configuration as the
        database. This is the default option when encryption_config is empty.
        For example, if the database is using `Customer_Managed_Encryption`,
        the backup will be using the same Cloud KMS key as the database.
      GOOGLE_DEFAULT_ENCRYPTION: Use Google default encryption.
      CUSTOMER_MANAGED_ENCRYPTION: Use customer managed encryption. If
        specified, `kms_key_name` must contain a valid Cloud KMS key.
    """
        ENCRYPTION_TYPE_UNSPECIFIED = 0
        USE_DATABASE_ENCRYPTION = 1
        GOOGLE_DEFAULT_ENCRYPTION = 2
        CUSTOMER_MANAGED_ENCRYPTION = 3
    encryptionType = _messages.EnumField('EncryptionTypeValueValuesEnum', 1)
    kmsKeyName = _messages.StringField(2)
    kmsKeyNames = _messages.StringField(3, repeated=True)