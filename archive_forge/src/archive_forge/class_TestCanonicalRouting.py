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
class TestCanonicalRouting(PecanTestCase):

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
        return TestApp(Pecan(RootController()))

    def test_root(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert b'index' in r.body

    def test_index(self):
        r = self.app_.get('/index')
        assert r.status_int == 200
        assert b'index' in r.body

    def test_broken_clients(self):
        r = self.app_.get('', status=302)
        assert r.status_int == 302
        assert r.location == 'http://localhost/'

    def test_sub_controller_with_trailing(self):
        r = self.app_.get('/sub/')
        assert r.status_int == 200
        assert b'subindex' in r.body

    def test_sub_controller_redirect(self):
        r = self.app_.get('/sub', status=302)
        assert r.status_int == 302
        assert r.location == 'http://localhost/sub/'

    def test_with_query_string(self):
        r = self.app_.get('/sub?foo=bar', status=302)
        assert r.status_int == 302
        assert r.location == 'http://localhost/sub/?foo=bar'

    def test_posts_fail(self):
        try:
            self.app_.post('/sub', dict(foo=1))
            raise Exception('Post should fail')
        except Exception as e:
            assert isinstance(e, RuntimeError)

    def test_with_args(self):
        r = self.app_.get('/arg/index/foo')
        assert r.status_int == 200
        assert r.body == b'foo'

    def test_accept_noncanonical(self):
        r = self.app_.get('/accept/')
        assert r.status_int == 200
        assert r.body == b'accept'

    def test_accept_noncanonical_no_trailing_slash(self):
        r = self.app_.get('/accept')
        assert r.status_int == 200
        assert r.body == b'accept'