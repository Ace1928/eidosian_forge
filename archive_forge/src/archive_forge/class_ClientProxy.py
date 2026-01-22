import functools
from dogpile.cache.backends import memcached as memcached_backend
from oslo_cache import _memcache_pool
from oslo_cache import exception
class ClientProxy(object):

    def __init__(self, client_pool):
        self.client_pool = client_pool

    def _run_method(self, __name, *args, **kwargs):
        with self.client_pool.acquire() as client:
            return getattr(client, __name)(*args, **kwargs)

    def __getattr__(self, name):
        return functools.partial(self._run_method, name)