import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
class _MemcacheClientPool(object):
    """An advanced memcached client pool that is eventlet safe."""

    def __init__(self, memcache_servers, arguments, **kwargs):
        from oslo_cache import _memcache_pool
        self._pool = _memcache_pool.MemcacheClientPool(memcache_servers, arguments, **kwargs)

    @contextlib.contextmanager
    def reserve(self):
        with self._pool.acquire() as client:
            yield client