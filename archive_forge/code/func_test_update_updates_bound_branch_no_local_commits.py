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
def test_update_updates_bound_branch_no_local_commits(self):
    master_tree = self.make_branch_and_tree('master')
    tree = self.make_branch_and_tree('tree')
    try:
        tree.branch.bind(master_tree.branch)
    except _mod_branch.BindingUnsupported:
        return
    foo = master_tree.commit('foo', allow_pointless=True)
    tree.update()
    self.assertEqual([foo], tree.get_parent_ids())
    self.assertEqual(foo, tree.branch.last_revision())