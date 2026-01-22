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
class TestReadPacksFile(TestCase):

    def test_read_packs(self):
        self.assertEqual(['pack-1.pack'], list(read_packs_file(BytesIO(b'P pack-1.pack\n'))))