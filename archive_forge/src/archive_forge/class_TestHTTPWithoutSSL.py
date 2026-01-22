import pytest
import urllib3
from dummyserver.testcase import HTTPDummyServerTestCase, HTTPSDummyServerTestCase
from ..test_no_ssl import TestWithoutSSL
class TestHTTPWithoutSSL(HTTPDummyServerTestCase, TestWithoutSSL):

    def test_simple(self):
        with urllib3.HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/')
            assert r.status == 200, r.data