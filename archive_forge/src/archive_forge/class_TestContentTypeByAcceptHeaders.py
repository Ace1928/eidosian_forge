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
class TestContentTypeByAcceptHeaders(PecanTestCase):

    @property
    def app_(self):
        """
        Test that content type is set appropriately based on Accept headers.
        """

        class RootController(object):

            @expose(content_type='text/html')
            @expose(content_type='application/json')
            def index(self, *args):
                return 'Foo'
        return TestApp(Pecan(RootController()))

    def test_missing_accept(self):
        r = self.app_.get('/', headers={'Accept': ''})
        assert r.status_int == 200
        assert r.content_type == 'text/html'

    def test_quality(self):
        r = self.app_.get('/', headers={'Accept': 'text/html,application/json;q=0.9,*/*;q=0.8'})
        assert r.status_int == 200
        assert r.content_type == 'text/html'
        r = self.app_.get('/', headers={'Accept': 'application/json,text/html;q=0.9,*/*;q=0.8'})
        assert r.status_int == 200
        assert r.content_type == 'application/json'

    def test_discarded_accept_parameters(self):
        r = self.app_.get('/', headers={'Accept': 'application/json;discard=me'})
        assert r.status_int == 200
        assert r.content_type == 'application/json'

    def test_file_extension_has_higher_precedence(self):
        r = self.app_.get('/index.html', headers={'Accept': 'application/json,text/html;q=0.9,*/*;q=0.8'})
        assert r.status_int == 200
        assert r.content_type == 'text/html'

    def test_not_acceptable(self):
        r = self.app_.get('/', headers={'Accept': 'application/xml'}, status=406)
        assert r.status_int == 406

    def test_accept_header_missing(self):
        r = self.app_.get('/')
        assert r.status_int == 200
        assert r.content_type == 'text/html'