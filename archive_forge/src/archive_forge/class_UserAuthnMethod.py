import logging
import time
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit
from saml2 import SAMLError
import saml2.cryptography.symmetric
from saml2.httputil import Redirect
from saml2.httputil import Response
from saml2.httputil import Unauthorized
from saml2.httputil import make_cookie
from saml2.httputil import parse_cookie
class UserAuthnMethod:

    def __init__(self, srv):
        self.srv = srv

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def authenticated_as(self, **kwargs):
        raise NotImplementedError

    def verify(self, **kwargs):
        raise NotImplementedError