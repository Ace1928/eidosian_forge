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
def test_with_file(self):
    child_ie = InventoryFile(b'bar', 'bar', b'bar')
    b = Blob.from_string(b'bla')
    t1 = directory_to_tree('', [child_ie], lambda p, x: b.id, {}, None, allow_empty=False)
    t2 = Tree()
    t2.add(b'bar', 33188, b.id)
    self.assertEqual(t1, t2)