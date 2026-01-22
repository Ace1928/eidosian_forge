import os
import shutil
import sys
import tempfile
import zlib
from hashlib import sha1
from io import BytesIO
from typing import Set
from dulwich.tests import TestCase
from ..errors import ApplyDeltaError, ChecksumMismatch
from ..file import GitFile
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit, Tree, hex_to_sha, sha_to_hex
from ..pack import (
from .utils import build_pack, make_object
class PackTests(TestCase):
    """Base class for testing packs."""

    def setUp(self):
        super().setUp()
        self.tempdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tempdir)
    datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../testdata/packs'))

    def get_pack_index(self, sha):
        """Returns a PackIndex from the datadir with the given sha."""
        return load_pack_index(os.path.join(self.datadir, 'pack-%s.idx' % sha.decode('ascii')))

    def get_pack_data(self, sha):
        """Returns a PackData object from the datadir with the given sha."""
        return PackData(os.path.join(self.datadir, 'pack-%s.pack' % sha.decode('ascii')))

    def get_pack(self, sha):
        return Pack(os.path.join(self.datadir, 'pack-%s' % sha.decode('ascii')))

    def assertSucceeds(self, func, *args, **kwargs):
        try:
            func(*args, **kwargs)
        except ChecksumMismatch as e:
            self.fail(e)