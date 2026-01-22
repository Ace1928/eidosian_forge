from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProvisioningQuota(_messages.Message):
    """A provisioning quota for a given project.

  Enums:
    AssetTypeValueValuesEnum: The asset type of this provisioning quota.

  Fields:
    assetType: The asset type of this provisioning quota.
    availableCount: The available count of the provisioning quota.
    gcpService: The gcp service of the provisioning quota.
    instanceQuota: Instance quota.
    location: The specific location of the provisioining quota.
    name: Output only. The name of the provisioning quota.
    networkBandwidth: Network bandwidth, Gbps
    serverCount: Server count.
    storageGib: Storage size (GB).
  """

    class AssetTypeValueValuesEnum(_messages.Enum):
        """The asset type of this provisioning quota.

    Values:
      ASSET_TYPE_UNSPECIFIED: The unspecified type.
      ASSET_TYPE_SERVER: The server asset type.
      ASSET_TYPE_STORAGE: The storage asset type.
      ASSET_TYPE_NETWORK: The network asset type.
    """
        ASSET_TYPE_UNSPECIFIED = 0
        ASSET_TYPE_SERVER = 1
        ASSET_TYPE_STORAGE = 2
        ASSET_TYPE_NETWORK = 3
    assetType = _messages.EnumField('AssetTypeValueValuesEnum', 1)
    availableCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    gcpService = _messages.StringField(3)
    instanceQuota = _messages.MessageField('InstanceQuota', 4)
    location = _messages.StringField(5)
    name = _messages.StringField(6)
    networkBandwidth = _messages.IntegerField(7)
    serverCount = _messages.IntegerField(8)
    storageGib = _messages.IntegerField(9)