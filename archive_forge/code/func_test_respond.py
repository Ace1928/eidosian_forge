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
def test_respond(self):
    self._req.nocache()
    self._req.respond(status=402, content_type='some/type', headers=[('X-Foo', 'foo'), ('X-Bar', 'bar')])
    self.assertEqual({('X-Foo', 'foo'), ('X-Bar', 'bar'), ('Content-Type', 'some/type'), ('Expires', 'Fri, 01 Jan 1980 00:00:00 GMT'), ('Pragma', 'no-cache'), ('Cache-Control', 'no-cache, max-age=0, must-revalidate')}, set(self._headers))
    self.assertEqual(402, self._status)