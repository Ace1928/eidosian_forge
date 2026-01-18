from requests import auth
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest
Resends a request with auth headers, if needed.