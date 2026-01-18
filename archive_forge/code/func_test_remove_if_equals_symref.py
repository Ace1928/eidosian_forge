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
def test_remove_if_equals_symref(self):
    self.assertFalse(self._refs.remove_if_equals(b'HEAD', b'42d06bd4b77fed026b154d16493e5deab78f02ec'))
    self.assertTrue(self._refs.remove_if_equals(b'refs/heads/master', b'42d06bd4b77fed026b154d16493e5deab78f02ec'))
    self.assertRaises(KeyError, lambda: self._refs[b'refs/heads/master'])
    self.assertRaises(KeyError, lambda: self._refs[b'HEAD'])
    self.assertEqual(b'ref: refs/heads/master', self._refs.read_loose_ref(b'HEAD'))
    self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'refs', b'heads', b'master.lock')))
    self.assertFalse(os.path.exists(os.path.join(self._refs.path, b'HEAD.lock')))