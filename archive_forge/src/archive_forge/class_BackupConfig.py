from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupConfig(_messages.Message):
    """BackupConfig defines the configuration of Backups created via this
  BackupPlan.

  Fields:
    allNamespaces: If True, include all namespaced resources
    encryptionKey: Optional. This defines a customer managed encryption key
      that will be used to encrypt the "config" portion (the Kubernetes
      resources) of Backups created via this plan. Default (empty): Config
      backup artifacts will not be encrypted.
    includeSecrets: Optional. This flag specifies whether Kubernetes Secret
      resources should be included when they fall into the scope of Backups.
      Default: False
    includeVolumeData: Optional. This flag specifies whether volume data
      should be backed up when PVCs are included in the scope of a Backup.
      Default: False
    permissiveMode: Optional. If false, Backups will fail when Backup for GKE
      detects Kubernetes configuration that is non-standard or requires
      additional setup to restore. Default: False
    selectedApplications: If set, include just the resources referenced by the
      listed ProtectedApplications.
    selectedNamespaces: If set, include just the resources in the listed
      namespaces.
  """
    allNamespaces = _messages.BooleanField(1)
    encryptionKey = _messages.MessageField('EncryptionKey', 2)
    includeSecrets = _messages.BooleanField(3)
    includeVolumeData = _messages.BooleanField(4)
    permissiveMode = _messages.BooleanField(5)
    selectedApplications = _messages.MessageField('NamespacedNames', 6)
    selectedNamespaces = _messages.MessageField('Namespaces', 7)