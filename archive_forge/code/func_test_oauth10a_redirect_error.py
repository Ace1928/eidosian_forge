import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
@unittest.skipIf(mock is None, 'mock package not present')
def test_oauth10a_redirect_error(self):
    with mock.patch.object(OAuth1ServerRequestTokenHandler, 'get') as get:
        get.side_effect = Exception('boom')
        with ExpectLog(app_log, 'Uncaught exception'):
            response = self.fetch('/oauth10a/client/login', follow_redirects=False)
        self.assertEqual(response.code, 500)