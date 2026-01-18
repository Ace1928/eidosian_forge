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
def test_setitem_packed(self):
    with open(os.path.join(self._refs.path, b'packed-refs'), 'w') as f:
        f.write('# pack-refs with: peeled fully-peeled sorted \n')
        f.write('42d06bd4b77fed026b154d16493e5deab78f02ec refs/heads/packed\n')
    self._refs[b'refs/heads/packed'] = b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8'
    packed_ref_path = os.path.join(self._refs.path, b'refs', b'heads', b'packed')
    with open(packed_ref_path, 'rb') as f:
        self.assertEqual(b'3ec9c43c84ff242e3ef4a9fc5bc111fd780a76a8', f.read()[:40])
    self.assertRaises(OSError, self._refs.__setitem__, b'refs/heads/packed/sub', b'42d06bd4b77fed026b154d16493e5deab78f02ec')
    self.assertEqual({b'refs/heads/packed': b'42d06bd4b77fed026b154d16493e5deab78f02ec'}, self._refs.get_packed_refs())