from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMemcacheV1UpgradeInstanceRequest(_messages.Message):
    """Request for UpgradeInstance.

  Enums:
    MemcacheVersionValueValuesEnum: Required. Specifies the target version of
      memcached engine to upgrade to.

  Fields:
    memcacheVersion: Required. Specifies the target version of memcached
      engine to upgrade to.
  """

    class MemcacheVersionValueValuesEnum(_messages.Enum):
        """Required. Specifies the target version of memcached engine to upgrade
    to.

    Values:
      MEMCACHE_VERSION_UNSPECIFIED: Memcache version is not specified by
        customer
      MEMCACHE_1_5: Memcached 1.5 version.
      MEMCACHE_1_6_15: Memcached 1.6.15 version.
    """
        MEMCACHE_VERSION_UNSPECIFIED = 0
        MEMCACHE_1_5 = 1
        MEMCACHE_1_6_15 = 2
    memcacheVersion = _messages.EnumField('MemcacheVersionValueValuesEnum', 1)