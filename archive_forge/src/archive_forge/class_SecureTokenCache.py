import contextlib
import hashlib
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystonemiddleware.auth_token import _exceptions as exc
from keystonemiddleware.auth_token import _memcache_crypt as memcache_crypt
from keystonemiddleware.i18n import _
class SecureTokenCache(TokenCache):
    """A token cache that stores tokens encrypted.

    A more secure version of TokenCache that will encrypt tokens before
    caching them.
    """

    def __init__(self, log, security_strategy, secret_key, **kwargs):
        super(SecureTokenCache, self).__init__(log, **kwargs)
        if not secret_key:
            msg = _('memcache_secret_key must be defined when a memcache_security_strategy is defined')
            raise exc.ConfigurationError(msg)
        if isinstance(security_strategy, str):
            security_strategy = security_strategy.encode('utf-8')
        if isinstance(secret_key, str):
            secret_key = secret_key.encode('utf-8')
        self._security_strategy = security_strategy
        self._secret_key = secret_key

    def _get_cache_key(self, token_id):
        context = memcache_crypt.derive_keys(token_id, self._secret_key, self._security_strategy)
        key = self._CACHE_KEY_TEMPLATE % memcache_crypt.get_cache_key(context)
        return (key, context)

    def _deserialize(self, data, context):
        try:
            return memcache_crypt.unprotect_data(context, data)
        except Exception:
            msg = 'Failed to decrypt/verify cache data'
            self._LOG.exception(msg)
        return None

    def _serialize(self, data, context):
        return memcache_crypt.protect_data(context, data)