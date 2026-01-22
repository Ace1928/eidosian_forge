from requests import auth
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest
def _handle_digest_auth_407(self, r, kwargs):
    self.proxy_auth = http_proxy_digest.HTTPProxyDigestAuth(username=self.proxy_username, password=self.proxy_password)
    try:
        self.auth.init_per_thread_state()
    except AttributeError:
        pass
    return self.proxy_auth.handle_407(r, **kwargs)