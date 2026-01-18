import os
import shutil
import stat
import sys
import tempfile
from contextlib import closing
from io import BytesIO
from unittest import skipUnless
from dulwich.tests import TestCase
from ..errors import NotTreeError
from ..index import commit_tree
from ..object_store import (
from ..objects import (
from ..pack import REF_DELTA, write_pack_objects
from ..protocol import DEPTH_INFINITE
from .utils import build_pack, make_object, make_tag
def test_rel_alternative_path(self):
    alternate_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, alternate_dir)
    alternate_store = DiskObjectStore(alternate_dir)
    b2 = make_object(Blob, data=b'yummy data')
    alternate_store.add_object(b2)
    store = DiskObjectStore(self.store_dir)
    self.assertRaises(KeyError, store.__getitem__, b2.id)
    store.add_alternate_path(os.path.relpath(alternate_dir, self.store_dir))
    self.assertEqual(list(alternate_store), list(store.alternates[0]))
    self.assertIn(b2.id, store)
    self.assertEqual(b2, store[b2.id])