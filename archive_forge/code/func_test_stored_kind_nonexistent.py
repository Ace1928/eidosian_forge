import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
def test_stored_kind_nonexistent(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.assertRaises(_mod_transport.NoSuchFile, tree.stored_kind, 'a')
    self.addCleanup(tree.unlock)
    self.build_tree(['tree/a'])
    self.assertRaises(_mod_transport.NoSuchFile, tree.stored_kind, 'a')
    tree.add(['a'])
    self.assertIs('file', tree.stored_kind('a'))