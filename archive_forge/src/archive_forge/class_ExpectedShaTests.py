import os
import shutil
import stat
from dulwich.objects import Blob, Tree
from ...branchbuilder import BranchBuilder
from ...bzr.inventory import InventoryDirectory, InventoryFile
from ...errors import NoSuchRevision
from ...graph import DictParentsProvider, Graph
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import SymlinkFeature
from ..cache import DictGitShaMap
from ..object_store import (BazaarObjectStore, LRUTreeCache,
class ExpectedShaTests(TestCase):

    def setUp(self):
        super().setUp()
        self.obj = Blob()
        self.obj.data = b'foo'

    def test_none(self):
        _check_expected_sha(None, self.obj)

    def test_hex(self):
        _check_expected_sha(self.obj.sha().hexdigest().encode('ascii'), self.obj)
        self.assertRaises(AssertionError, _check_expected_sha, b'0' * 40, self.obj)

    def test_binary(self):
        _check_expected_sha(self.obj.sha().digest(), self.obj)
        self.assertRaises(AssertionError, _check_expected_sha, b'x' * 20, self.obj)