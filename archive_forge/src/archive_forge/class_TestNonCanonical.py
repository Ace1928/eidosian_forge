import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
class TestNonCanonical(PecanTestCase):

    @property
    def app_(self):

        class ArgSubController(object):

            @expose()
            def index(self, arg):
                return arg

        class AcceptController(object):

            @accept_noncanonical
            @expose()
            def index(self):
                return 'accept'

        class SubController(object):

            @expose()
            def index(self, **kw):
                return 'subindex'

        class RootController(object):

            @expose()
            def index(self):
                return 'index'
            sub = SubController()
            arg = ArgSubController()
            accept = AcceptController()
        return TestApp(Pecan(RootController(), force_canonical=False))

    def test_index(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert b'index' in r.body

    def test_subcontroller(self):
        r = self.app_.get('/sub')
        assert r.status_int == 200
        assert b'subindex' in r.body

    def test_subcontroller_with_kwargs(self):
        r = self.app_.post('/sub', dict(foo=1))
        assert r.status_int == 200
        assert b'subindex' in r.body

    def test_sub_controller_with_trailing(self):
        r = self.app_.get('/sub/')
        assert r.status_int == 200
        assert b'subindex' in r.body

    def test_proxy(self):

        class RootController(object):

            @expose()
            def index(self):
                request.testing = True
                assert request.testing is True
                del request.testing
                assert hasattr(request, 'testing') is False
                return '/'
        app = TestApp(make_app(RootController(), debug=True))
        r = app.get('/')
        assert r.status_int == 200

    def test_app_wrap(self):

        class RootController(object):
            pass
        wrapped_apps = []

        def wrap(app):
            wrapped_apps.append(app)
            return app
        make_app(RootController(), wrap_app=wrap, debug=True)
        assert len(wrapped_apps) == 1