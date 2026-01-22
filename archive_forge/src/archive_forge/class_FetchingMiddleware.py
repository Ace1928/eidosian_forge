import datetime
from unittest import mock
import uuid
from keystoneauth1 import fixture
import testtools
import webob
from keystonemiddleware import auth_token
from keystonemiddleware.auth_token import _request
class FetchingMiddleware(auth_token.BaseAuthProtocol):

    def __init__(self, app, token_dict={}, **kwargs):
        super(FetchingMiddleware, self).__init__(app, **kwargs)
        self.token_dict = token_dict

    def fetch_token(self, token, **kwargs):
        try:
            return self.token_dict[token]
        except KeyError:
            raise auth_token.InvalidToken()