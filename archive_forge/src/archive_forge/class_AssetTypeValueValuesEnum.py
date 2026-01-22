from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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