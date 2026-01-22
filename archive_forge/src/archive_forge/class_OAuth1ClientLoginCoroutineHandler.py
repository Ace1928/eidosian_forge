import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class OAuth1ClientLoginCoroutineHandler(OAuth1ClientLoginHandler):
    """Replaces OAuth1ClientLoginCoroutineHandler's get() with a coroutine."""

    @gen.coroutine
    def get(self):
        if self.get_argument('oauth_token', None):
            try:
                yield self.get_authenticated_user()
            except Exception as e:
                self.set_status(503)
                self.write('got exception: %s' % e)
        else:
            yield self.authorize_redirect()