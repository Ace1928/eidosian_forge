from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EncryptionValue(_messages.Message):
    """Encryption configuration for a bucket.

    Fields:
      defaultKmsKeyName: A Cloud KMS key that will be used to encrypt objects
        inserted into this bucket, if no encryption method is specified.
    """
    defaultKmsKeyName = _messages.StringField(1)