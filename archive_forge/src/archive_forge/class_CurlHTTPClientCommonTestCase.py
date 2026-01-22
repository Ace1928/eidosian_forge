from hashlib import md5
import unittest
from tornado.escape import utf8
from tornado.testing import AsyncHTTPTestCase
from tornado.test import httpclient_test
from tornado.web import Application, RequestHandler
@unittest.skipIf(pycurl is None, 'pycurl module not present')
class CurlHTTPClientCommonTestCase(httpclient_test.HTTPClientCommonTestCase):

    def get_http_client(self):
        client = CurlAsyncHTTPClient(defaults=dict(allow_ipv6=False))
        self.assertTrue(isinstance(client, CurlAsyncHTTPClient))
        return client