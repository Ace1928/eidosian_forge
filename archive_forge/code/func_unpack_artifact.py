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
def unpack_artifact(environ):
    if environ['REQUEST_METHOD'] == 'GET':
        _dict = unpack_redirect(environ)
    elif environ['REQUEST_METHOD'] == 'POST':
        _dict = unpack_post(environ)
    else:
        _dict = None
    return _dict