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
def test_remove_parent(self):
    self._refs[b'refs/heads/foo/bar'] = b'df6800012397fb85c56e7418dd4eb9405dee075c'
    del self._refs[b'refs/heads/foo/bar']
    ref_file = os.path.join(self._refs.path, b'refs', b'heads', b'foo', b'bar')
    self.assertFalse(os.path.exists(ref_file))
    ref_file = os.path.join(self._refs.path, b'refs', b'heads', b'foo')
    self.assertFalse(os.path.exists(ref_file))
    ref_file = os.path.join(self._refs.path, b'refs', b'heads')
    self.assertTrue(os.path.exists(ref_file))
    self._refs[b'refs/heads/foo'] = b'df6800012397fb85c56e7418dd4eb9405dee075c'