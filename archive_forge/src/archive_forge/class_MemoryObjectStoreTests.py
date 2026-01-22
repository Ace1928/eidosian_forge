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
class MemoryObjectStoreTests(ObjectStoreTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.store = MemoryObjectStore()

    def test_add_pack(self):
        o = MemoryObjectStore()
        f, commit, abort = o.add_pack()
        try:
            b = make_object(Blob, data=b'more yummy data')
            write_pack_objects(f.write, [(b, None)])
        except BaseException:
            abort()
            raise
        else:
            commit()

    def test_add_pack_emtpy(self):
        o = MemoryObjectStore()
        f, commit, abort = o.add_pack()
        commit()

    def test_add_thin_pack(self):
        o = MemoryObjectStore()
        blob = make_object(Blob, data=b'yummy data')
        o.add_object(blob)
        f = BytesIO()
        entries = build_pack(f, [(REF_DELTA, (blob.id, b'more yummy data'))], store=o)
        o.add_thin_pack(f.read, None)
        packed_blob_sha = sha_to_hex(entries[0][3])
        self.assertEqual((Blob.type_num, b'more yummy data'), o.get_raw(packed_blob_sha))

    def test_add_thin_pack_empty(self):
        o = MemoryObjectStore()
        f = BytesIO()
        entries = build_pack(f, [], store=o)
        self.assertEqual([], entries)
        o.add_thin_pack(f.read, None)