import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class TwitterClientHandler(RequestHandler, TwitterMixin):

    def initialize(self, test):
        self._OAUTH_REQUEST_TOKEN_URL = test.get_url('/oauth1/server/request_token')
        self._OAUTH_ACCESS_TOKEN_URL = test.get_url('/twitter/server/access_token')
        self._OAUTH_AUTHORIZE_URL = test.get_url('/oauth1/server/authorize')
        self._OAUTH_AUTHENTICATE_URL = test.get_url('/twitter/server/authenticate')
        self._TWITTER_BASE_URL = test.get_url('/twitter/api')

    def get_auth_http_client(self):
        return self.settings['http_client']