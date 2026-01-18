import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_twitter_authenticate_redirect(self):
    response = self.fetch('/twitter/client/authenticate', follow_redirects=False)
    self.assertEqual(response.code, 302)
    self.assertTrue(response.headers['Location'].endswith('/twitter/server/authenticate?oauth_token=zxcv'), response.headers['Location'])
    self.assertTrue('_oauth_request_token="enhjdg==|MTIzNA=="' in response.headers['Set-Cookie'], response.headers['Set-Cookie'])