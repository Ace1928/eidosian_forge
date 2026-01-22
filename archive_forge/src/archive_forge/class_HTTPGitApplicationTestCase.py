import gzip
import os
import re
from io import BytesIO
from typing import Type
from dulwich.tests import TestCase
from ..object_store import MemoryObjectStore
from ..objects import Blob
from ..repo import BaseRepo, MemoryRepo
from ..server import DictBackend
from ..web import (
from .utils import make_object, make_tag
class HTTPGitApplicationTestCase(TestCase):

    def setUp(self):
        super().setUp()
        self._app = HTTPGitApplication('backend')
        self._environ = {'PATH_INFO': '/foo', 'REQUEST_METHOD': 'GET'}

    def _test_handler(self, req, backend, mat):
        self.assertEqual(self._environ, req.environ)
        self.assertEqual('backend', backend)
        self.assertEqual('/foo', mat.group(0))
        return 'output'

    def _add_handler(self, app):
        req = self._environ['REQUEST_METHOD']
        app.services = {(req, re.compile('/foo$')): self._test_handler}

    def test_call(self):
        self._add_handler(self._app)
        self.assertEqual('output', self._app(self._environ, None))

    def test_fallback_app(self):

        def test_app(environ, start_response):
            return 'output'
        app = HTTPGitApplication('backend', fallback_app=test_app)
        self.assertEqual('output', app(self._environ, None))