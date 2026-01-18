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
def test_send_file_not_found(self):
    list(send_file(self._req, None, 'text/plain'))
    self.assertEqual(HTTP_NOT_FOUND, self._status)