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
def test_two_missing(self):
    self.assertEqual({'a', 'b'}, self._find_missing({'a': ['b'], 'b': ['c']}, ['a'], ['c']))