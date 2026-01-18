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
def test_read_ref(self):
    self.assertEqual(b'ref: refs/heads/master', self._refs.read_ref(b'HEAD'))
    self.assertEqual(b'42d06bd4b77fed026b154d16493e5deab78f02ec', self._refs.read_ref(b'refs/heads/packed'))
    self.assertEqual(None, self._refs.read_ref(b'nonexistent'))