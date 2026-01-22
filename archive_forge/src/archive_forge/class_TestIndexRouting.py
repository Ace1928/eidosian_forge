import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
class TestIndexRouting(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def index(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return 'Hello, World!'
        return TestApp(Pecan(RootController(), use_context_locals=False))

    def test_empty_root(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert r.body == b'Hello, World!'

    def test_index(self):
        r = self.app_.get('/index')
        assert r.status_int == 200
        assert r.body == b'Hello, World!'

    def test_index_html(self):
        r = self.app_.get('/index.html')
        assert r.status_int == 200
        assert r.body == b'Hello, World!'