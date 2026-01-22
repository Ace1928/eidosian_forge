import hashlib
import hmac
import os
import six
from ._cookiejar import SimpleCookieJar
from ._exceptions import *
from ._http import *
from ._logging import *
from ._socket import *
def _get_resp_headers(sock, success_statuses=(101, 301, 302, 303)):
    status, resp_headers, status_message = read_headers(sock)
    if status not in success_statuses:
        raise WebSocketBadStatusException('Handshake status %d %s', status, status_message, resp_headers)
    return (status, resp_headers)