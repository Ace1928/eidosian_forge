import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_twitter_show_user(self):
    response = self.fetch('/twitter/client/show_user?name=somebody')
    response.rethrow()
    self.assertEqual(json_decode(response.body), {'name': 'Somebody', 'screen_name': 'somebody'})