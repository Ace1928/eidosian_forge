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
def test_read_alternate_paths(self):
    store = DiskObjectStore(self.store_dir)
    abs_path = os.path.abspath(os.path.normpath('/abspath'))
    store.add_alternate_path(abs_path)
    self.assertEqual(set(store._read_alternate_paths()), {abs_path})
    store.add_alternate_path('relative-path')
    self.assertIn(os.path.join(store.path, 'relative-path'), set(store._read_alternate_paths()))
    store.add_alternate_path('# comment')
    for alt_path in store._read_alternate_paths():
        self.assertNotIn('#', alt_path)