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
def test_update_returns_conflict_count(self):
    self.requireBranchReference()
    wt = self.make_branch_and_tree('tree')
    self.build_tree(['checkout/', 'tree/file'])
    checkout = bzrdir.BzrDirMetaFormat1().initialize('checkout')
    checkout.set_branch_reference(wt.branch)
    old_tree = self.workingtree_format.initialize(checkout)
    wt.add('file')
    a = wt.commit('A')
    self.build_tree(['checkout/file'])
    old_tree.add('file')
    self.assertEqual(1, old_tree.update())
    self.assertEqual([a], old_tree.get_parent_ids())