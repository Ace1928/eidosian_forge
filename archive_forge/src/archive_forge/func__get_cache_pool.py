import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
def _get_cache_pool(self, cache):
    if cache:
        return _EnvCachePool(cache)
    elif self._use_advanced_pool and self._memcached_servers:
        return _MemcacheClientPool(self._memcached_servers, self._arguments, **self._memcache_pool_options)
    else:
        if not self._use_advanced_pool:
            self._LOG.warning('Using the eventlet-unsafe cache pool is deprecated.It is recommended to use eventlet-safe cache poolimplementation from oslo.cache. This can be enabledthrough config option memcache_use_advanced_pool = True')
        return _CachePool(self._memcached_servers, self._LOG)