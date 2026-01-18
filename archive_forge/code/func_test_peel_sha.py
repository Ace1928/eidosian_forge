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
def test_peel_sha(self):
    self.store.add_object(testobject)
    tag1 = self.make_tag(b'1', testobject)
    tag2 = self.make_tag(b'2', testobject)
    tag3 = self.make_tag(b'3', testobject)
    for obj in [testobject, tag1, tag2, tag3]:
        self.assertEqual((obj, testobject), peel_sha(self.store, obj.id))