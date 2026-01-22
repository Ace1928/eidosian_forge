from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStorageConfig(_messages.Message):
    """BareMetalStorageConfig specifies the cluster storage configuration.

  Fields:
    lvpNodeMountsConfig: Required. Specifies the config for local
      PersistentVolumes backed by mounted node disks. These disks need to be
      formatted and mounted by the user, which can be done before or after
      cluster creation.
    lvpShareConfig: Required. Specifies the config for local PersistentVolumes
      backed by subdirectories in a shared filesystem. These subdirectores are
      automatically created during cluster creation.
  """
    lvpNodeMountsConfig = _messages.MessageField('BareMetalLvpConfig', 1)
    lvpShareConfig = _messages.MessageField('BareMetalLvpShareConfig', 2)