from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class InstancesReencryptRequest(_messages.Message):
    """Database Instance reencrypt request.

  Fields:
    backupReencryptionConfig: Configuration specific to backup re-encryption
  """
    backupReencryptionConfig = _messages.MessageField('BackupReencryptionConfig', 1)