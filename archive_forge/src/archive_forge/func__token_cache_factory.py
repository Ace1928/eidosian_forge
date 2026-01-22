from the request environment and it's identified by the ``swift.cache`` key.
import copy
import re
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import loading
from keystoneauth1.loading import session as session_loading
import oslo_cache
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.auth_token import _auth
from keystonemiddleware.auth_token import _base
from keystonemiddleware.auth_token import _cache
from keystonemiddleware.auth_token import _exceptions as ksm_exceptions
from keystonemiddleware.auth_token import _identity
from keystonemiddleware.auth_token import _opts
from keystonemiddleware.auth_token import _request
from keystonemiddleware.auth_token import _user_plugin
from keystonemiddleware.i18n import _
def _token_cache_factory(self):
    security_strategy = self._conf.get('memcache_security_strategy')
    cache_kwargs = dict(cache_time=int(self._conf.get('token_cache_time')), env_cache_name=self._conf.get('cache'), memcached_servers=self._conf.get('memcached_servers'), use_advanced_pool=self._conf.get('memcache_use_advanced_pool'), dead_retry=self._conf.get('memcache_pool_dead_retry'), maxsize=self._conf.get('memcache_pool_maxsize'), unused_timeout=self._conf.get('memcache_pool_unused_timeout'), conn_get_timeout=self._conf.get('memcache_pool_conn_get_timeout'), socket_timeout=self._conf.get('memcache_pool_socket_timeout'))
    if security_strategy.lower() != 'none':
        secret_key = self._conf.get('memcache_secret_key')
        return _cache.SecureTokenCache(self.log, security_strategy, secret_key, **cache_kwargs)
    else:
        return _cache.TokenCache(self.log, **cache_kwargs)