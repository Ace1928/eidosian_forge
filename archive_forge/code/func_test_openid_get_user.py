import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
def test_openid_get_user(self):
    response = self.fetch('/openid/client/login?openid.mode=blah&openid.ns.ax=http://openid.net/srv/ax/1.0&openid.ax.type.email=http://axschema.org/contact/email&openid.ax.value.email=foo@example.com')
    response.rethrow()
    parsed = json_decode(response.body)
    self.assertEqual(parsed['email'], 'foo@example.com')