import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_twitter_get_user(self):
    response = self.fetch('/twitter/client/login?oauth_token=zxcv', headers={'Cookie': '_oauth_request_token=enhjdg==|MTIzNA=='})
    response.rethrow()
    parsed = json_decode(response.body)
    self.assertEqual(parsed, {'access_token': {'key': 'hjkl', 'screen_name': 'foo', 'secret': 'vbnm'}, 'name': 'Foo', 'screen_name': 'foo', 'username': 'foo'})