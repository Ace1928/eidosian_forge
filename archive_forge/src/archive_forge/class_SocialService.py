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
class SocialService(UserAuthnMethod):

    def __init__(self, social):
        UserAuthnMethod.__init__(self, None)
        self.social = social

    def __call__(self, server_env, cookie=None, sid='', query='', **kwargs):
        return self.social.begin(server_env, cookie, sid, query)

    def callback(self, server_env, cookie=None, sid='', query='', **kwargs):
        return self.social.callback(server_env, cookie, sid, query, **kwargs)