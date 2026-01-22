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
class TestThreadingLocalUsage(PecanTestCase):

    @property
    def root(self):

        class RootController(object):

            @expose()
            def index(self, req, resp):
                assert isinstance(req, webob.BaseRequest)
                assert isinstance(resp, webob.Response)
                return 'Hello, World!'

            @expose()
            def warning(self):
                return 'This should be unroutable because (req, resp) are not arguments.  It should raise a TypeError.'

            @expose(generic=True)
            def generic(self):
                return 'This should be unroutable because (req, resp) are not arguments.  It should raise a TypeError.'

            @generic.when(method='PUT')
            def generic_put(self, _id):
                return 'This should be unroutable because (req, resp) are not arguments.  It should raise a TypeError.'
        return RootController

    def test_locals_are_not_used(self):
        with mock.patch('threading.local', side_effect=AssertionError()):
            app = TestApp(Pecan(self.root(), use_context_locals=False))
            r = app.get('/')
            assert r.status_int == 200
            assert r.body == b'Hello, World!'
            self.assertRaises(AssertionError, Pecan, self.root)

    def test_threadlocal_argument_warning(self):
        with mock.patch('threading.local', side_effect=AssertionError()):
            app = TestApp(Pecan(self.root(), use_context_locals=False))
            self.assertRaises(TypeError, app.get, '/warning/')

    def test_threadlocal_argument_warning_on_generic(self):
        with mock.patch('threading.local', side_effect=AssertionError()):
            app = TestApp(Pecan(self.root(), use_context_locals=False))
            self.assertRaises(TypeError, app.get, '/generic/')

    def test_threadlocal_argument_warning_on_generic_delegate(self):
        with mock.patch('threading.local', side_effect=AssertionError()):
            app = TestApp(Pecan(self.root(), use_context_locals=False))
            self.assertRaises(TypeError, app.put, '/generic/')