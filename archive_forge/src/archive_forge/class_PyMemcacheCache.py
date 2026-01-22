import re
import time
from django.core.cache.backends.base import (
from django.utils.functional import cached_property
class PyMemcacheCache(BaseMemcachedCache):
    """An implementation of a cache binding using pymemcache."""

    def __init__(self, server, params):
        import pymemcache.serde
        super().__init__(server, params, library=pymemcache, value_not_found_exception=KeyError)
        self._class = self._lib.HashClient
        self._options = {'allow_unicode_keys': True, 'default_noreply': False, 'serde': pymemcache.serde.pickle_serde, **self._options}