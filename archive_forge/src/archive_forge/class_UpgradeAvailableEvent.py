from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpgradeAvailableEvent(_messages.Message):
    """UpgradeAvailableEvent is a notification sent to customers when a new
  available version is released.

  Enums:
    ResourceTypeValueValuesEnum: The resource type of the release version.

  Fields:
    releaseChannel: The release channel of the version. If empty, it means a
      non-channel release.
    resource: Optional relative path to the resource. For example, the
      relative path of the node pool.
    resourceType: The resource type of the release version.
    version: The release version available for upgrade.
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """The resource type of the release version.

    Values:
      UPGRADE_RESOURCE_TYPE_UNSPECIFIED: Default value. This shouldn't be
        used.
      MASTER: Master / control plane
      NODE_POOL: Node pool
    """
        UPGRADE_RESOURCE_TYPE_UNSPECIFIED = 0
        MASTER = 1
        NODE_POOL = 2
    releaseChannel = _messages.MessageField('ReleaseChannel', 1)
    resource = _messages.StringField(2)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 3)
    version = _messages.StringField(4)