import re
import time
from django.core.cache.backends.base import (
from django.utils.functional import cached_property
class PyLibMCCache(BaseMemcachedCache):
    """An implementation of a cache binding using pylibmc"""

    def __init__(self, server, params):
        import pylibmc
        super().__init__(server, params, library=pylibmc, value_not_found_exception=pylibmc.NotFound)

    @property
    def client_servers(self):
        output = []
        for server in self._servers:
            output.append(server.removeprefix('unix:'))
        return output

    def touch(self, key, timeout=DEFAULT_TIMEOUT, version=None):
        key = self.make_and_validate_key(key, version=version)
        if timeout == 0:
            return self._cache.delete(key)
        return self._cache.touch(key, self.get_backend_timeout(timeout))

    def close(self, **kwargs):
        pass