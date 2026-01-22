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
def _test_call(self, orig, zstream, zlength):
    self._add_handler(self._app.app)
    self.assertLess(zlength, len(orig))
    self.assertEqual(self._environ['HTTP_CONTENT_ENCODING'], 'gzip')
    self._environ['CONTENT_LENGTH'] = zlength
    self._environ['wsgi.input'] = zstream
    self._app(self._environ, None)
    buf = self._environ['wsgi.input']
    self.assertIsNot(buf, zstream)
    buf.seek(0)
    self.assertEqual(orig, buf.read())
    self.assertIs(None, self._environ.get('CONTENT_LENGTH'))
    self.assertNotIn('HTTP_CONTENT_ENCODING', self._environ)