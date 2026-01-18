import cgi
import hashlib
import hmac
from http.cookies import SimpleCookie
import logging
import time
from typing import Optional
from urllib.parse import parse_qs
from urllib.parse import quote
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import SAMLError
from saml2 import time_util
def unpack_redirect(environ):
    if 'QUERY_STRING' in environ:
        _qs = environ['QUERY_STRING']
        return {k: v[0] for k, v in parse_qs(_qs).items()}
    else:
        return None