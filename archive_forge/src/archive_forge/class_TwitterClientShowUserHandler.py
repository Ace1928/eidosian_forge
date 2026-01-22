import unittest
from tornado.auth import (
from tornado.escape import json_decode
from tornado import gen
from tornado.httpclient import HTTPClientError
from tornado.httputil import url_concat
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, ExpectLog
from tornado.web import RequestHandler, Application, HTTPError
class TwitterClientShowUserHandler(TwitterClientHandler):

    @gen.coroutine
    def get(self):
        try:
            response = (yield self.twitter_request('/users/show/%s' % self.get_argument('name'), access_token=dict(key='hjkl', secret='vbnm')))
        except HTTPClientError:
            self.set_status(500)
            self.finish('error from twitter request')
        else:
            self.finish(response)