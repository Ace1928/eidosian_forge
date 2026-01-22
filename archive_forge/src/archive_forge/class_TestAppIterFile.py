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
class TestAppIterFile(PecanTestCase):

    @property
    def app_(self):

        class RootController(object):

            @expose()
            def index(self):
                body = BytesIO(b'Hello, World!')
                response.body_file = body

            @expose()
            def empty(self):
                body = BytesIO(b'')
                response.body_file = body
        return TestApp(Pecan(RootController()))

    def test_body_generator(self):
        r = self.app_.get('/')
        self.assertEqual(r.status_int, 200)
        assert r.body == b'Hello, World!'

    def test_empty_body_generator(self):
        r = self.app_.get('/empty')
        self.assertEqual(r.status_int, 204)
        assert len(r.body) == 0