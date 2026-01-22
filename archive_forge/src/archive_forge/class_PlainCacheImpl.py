import os
from mako.cache import CacheImpl
from mako.cache import register_plugin
from mako.template import Template
from .assertions import eq_
from .config import config
class PlainCacheImpl(CacheImpl):
    """Simple memory cache impl so that tests which
    use caching can run without beaker."""

    def __init__(self, cache):
        self.cache = cache
        self.data = {}

    def get_or_create(self, key, creation_function, **kw):
        if key in self.data:
            return self.data[key]
        else:
            self.data[key] = data = creation_function(**kw)
            return data

    def put(self, key, value, **kw):
        self.data[key] = value

    def get(self, key, **kw):
        return self.data[key]

    def invalidate(self, key, **kw):
        del self.data[key]