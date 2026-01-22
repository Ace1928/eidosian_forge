import os
import shutil
import sys
import tempfile
from io import BytesIO
from typing import Dict, List
from dulwich.tests import TestCase
from ..errors import (
from ..object_store import MemoryObjectStore
from ..objects import Tree
from ..protocol import ZERO_SHA, format_capability_line
from ..repo import MemoryRepo, Repo
from ..server import (
from .utils import make_commit, make_tag
class ProtocolGraphWalkerEmptyTestCase(TestCase):

    def setUp(self):
        super().setUp()
        self._repo = MemoryRepo.init_bare([], {})
        backend = DictBackend({b'/': self._repo})
        self._walker = _ProtocolGraphWalker(TestUploadPackHandler(backend, [b'/', b'host=lolcats'], TestProto()), self._repo.object_store, self._repo.get_peeled, self._repo.refs.get_symrefs)

    def test_empty_repository(self):
        self._walker.proto.set_output([])
        self.assertRaises(HangupException, self._walker.determine_wants, {})
        self.assertEqual(None, self._walker.proto.get_received_line())
        self._walker.proto.set_output([None])
        self.assertEqual([], self._walker.determine_wants({}))
        self.assertEqual(None, self._walker.proto.get_received_line())