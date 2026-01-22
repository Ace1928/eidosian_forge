import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class TwitterClientAuthenticateHandler(TwitterClientHandler):

    @gen.coroutine
    def get(self):
        if self.get_argument('oauth_token', None):
            user = (yield self.get_authenticated_user())
            if user is None:
                raise Exception('user is None')
            self.finish(user)
            return
        yield self.authenticate_redirect()