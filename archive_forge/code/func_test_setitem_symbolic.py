import os
import sys
import tempfile
from io import BytesIO
from typing import ClassVar, Dict
from dulwich import errors
from dulwich.tests import SkipTest, TestCase
from ..file import GitFile
from ..objects import ZERO_SHA
from ..refs import (
from ..repo import Repo
from .utils import open_repo, tear_down_repo
def test_setitem_symbolic(self):
    ones = b'1' * 40
    self._refs[b'HEAD'] = ones
    self.assertEqual(ones, self._refs[b'HEAD'])
    f = open(os.path.join(self._refs.path, b'HEAD'), 'rb')
    v = next(iter(f)).rstrip(b'\n\r')
    f.close()
    self.assertEqual(b'ref: refs/heads/master', v)
    f = open(os.path.join(self._refs.path, b'refs', b'heads', b'master'), 'rb')
    self.assertEqual(ones, f.read()[:40])
    f.close()