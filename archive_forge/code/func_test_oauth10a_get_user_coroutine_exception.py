import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_oauth10a_get_user_coroutine_exception(self):
    response = self.fetch('/oauth10a/client/login_coroutine?oauth_token=zxcv&fail_in_get_user=true', headers={'Cookie': '_oauth_request_token=enhjdg==|MTIzNA=='})
    self.assertEqual(response.code, 503)