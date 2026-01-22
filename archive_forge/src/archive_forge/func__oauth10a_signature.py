import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def _oauth10a_signature(consumer_token: Dict[str, Any], method: str, url: str, parameters: Dict[str, Any]={}, token: Optional[Dict[str, Any]]=None) -> bytes:
    """Calculates the HMAC-SHA1 OAuth 1.0a signature for the given request.

    See http://oauth.net/core/1.0a/#signing_process
    """
    parts = urllib.parse.urlparse(url)
    scheme, netloc, path = parts[:3]
    normalized_url = scheme.lower() + '://' + netloc.lower() + path
    base_elems = []
    base_elems.append(method.upper())
    base_elems.append(normalized_url)
    base_elems.append('&'.join(('%s=%s' % (k, _oauth_escape(str(v))) for k, v in sorted(parameters.items()))))
    base_string = '&'.join((_oauth_escape(e) for e in base_elems))
    key_elems = [escape.utf8(urllib.parse.quote(consumer_token['secret'], safe='~'))]
    key_elems.append(escape.utf8(urllib.parse.quote(token['secret'], safe='~') if token else ''))
    key = b'&'.join(key_elems)
    hash = hmac.new(key, escape.utf8(base_string), hashlib.sha1)
    return binascii.b2a_base64(hash.digest())[:-1]