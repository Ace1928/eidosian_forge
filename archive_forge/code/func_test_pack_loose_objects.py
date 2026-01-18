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
def test_pack_loose_objects(self):
    b1 = make_object(Blob, data=b'yummy data')
    self.store.add_object(b1)
    b2 = make_object(Blob, data=b'more yummy data')
    self.store.add_object(b2)
    b3 = make_object(Blob, data=b'even more yummy data')
    b4 = make_object(Blob, data=b'and more yummy data')
    self.store.add_objects([(b3, None), (b4, None)])
    self.assertEqual({b1.id, b2.id, b3.id, b4.id}, set(self.store))
    self.assertEqual(1, len(self.store.packs))
    self.assertEqual(2, self.store.pack_loose_objects())
    self.assertNotEqual([], list(self.store.packs))
    self.assertEqual(0, self.store.pack_loose_objects())