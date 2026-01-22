from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.cache import cache_update_ops
def AddCacheUpdater(self, cache_updater):
    """Adds a cache_updater to the display info, newer values takes precedence.

    The cache updater is called to update the resource cache for CreateCommand,
    DeleteCommand and ListCommand commands.

    Args:
      cache_updater: A resource_cache.Updater class that will be instantiated
        and called to update the cache to reflect the resources returned by the
        calling command. None disables cache update.
    """
    self._cache_updater = cache_updater or cache_update_ops.NoCacheUpdater