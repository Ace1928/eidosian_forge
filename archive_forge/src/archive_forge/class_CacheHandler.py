from django.core import signals
from django.core.cache.backends.base import (
from django.utils.connection import BaseConnectionHandler, ConnectionProxy
from django.utils.module_loading import import_string
class CacheHandler(BaseConnectionHandler):
    settings_name = 'CACHES'
    exception_class = InvalidCacheBackendError

    def create_connection(self, alias):
        params = self.settings[alias].copy()
        backend = params.pop('BACKEND')
        location = params.pop('LOCATION', '')
        try:
            backend_cls = import_string(backend)
        except ImportError as e:
            raise InvalidCacheBackendError("Could not find backend '%s': %s" % (backend, e)) from e
        return backend_cls(location, params)