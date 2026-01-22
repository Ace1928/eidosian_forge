import re
import ssl
import urllib.parse
import dogpile.cache
from dogpile.cache import api
from dogpile.cache import proxy
from dogpile.cache import util
from oslo_log import log
from oslo_utils import importutils
from oslo_cache._i18n import _
from oslo_cache import _opts
from oslo_cache import exception
class _DebugProxy(proxy.ProxyBackend):
    """Extra Logging ProxyBackend."""

    def get(self, key):
        value = self.proxied.get(key)
        _LOG.debug('CACHE_GET: Key: "%(key)r" Value: "%(value)r"', {'key': key, 'value': value})
        return value

    def get_multi(self, keys):
        values = self.proxied.get_multi(keys)
        _LOG.debug('CACHE_GET_MULTI: "%(keys)r" Values: "%(values)r"', {'keys': keys, 'values': values})
        return values

    def set(self, key, value):
        _LOG.debug('CACHE_SET: Key: "%(key)r" Value: "%(value)r"', {'key': key, 'value': value})
        return self.proxied.set(key, value)

    def set_multi(self, keys):
        _LOG.debug('CACHE_SET_MULTI: "%r"', keys)
        self.proxied.set_multi(keys)

    def delete(self, key):
        self.proxied.delete(key)
        _LOG.debug('CACHE_DELETE: "%r"', key)

    def delete_multi(self, keys):
        _LOG.debug('CACHE_DELETE_MULTI: "%r"', keys)
        self.proxied.delete_multi(keys)