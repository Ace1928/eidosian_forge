import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class GoogleOAuth2UserinfoHandler(RequestHandler):

    def get(self):
        assert self.get_argument('access_token') == 'fake-access-token'
        self.finish({'name': 'Foo', 'email': 'foo@example.com'})