import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class GoogleLoginHandler(RequestHandler, GoogleOAuth2Mixin):

    def initialize(self, test):
        self.test = test
        self._OAUTH_REDIRECT_URI = test.get_url('/client/login')
        self._OAUTH_AUTHORIZE_URL = test.get_url('/google/oauth2/authorize')
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url('/google/oauth2/token')

    @gen.coroutine
    def get(self):
        code = self.get_argument('code', None)
        if code is not None:
            access = (yield self.get_authenticated_user(self._OAUTH_REDIRECT_URI, code))
            user = (yield self.oauth2_request(self.test.get_url('/google/oauth2/userinfo'), access_token=access['access_token']))
            user['access_token'] = access['access_token']
            self.write(user)
        else:
            self.authorize_redirect(redirect_uri=self._OAUTH_REDIRECT_URI, client_id=self.settings['google_oauth']['key'], scope=['profile', 'email'], response_type='code', extra_params={'prompt': 'select_account'})