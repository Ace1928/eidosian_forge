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
def test_post_build_tree_hook(self):
    calls = []

    def track_post_build_tree(tree):
        calls.append(tree.last_revision())
    source = self.make_branch_and_tree('source')
    a = source.commit('a', allow_pointless=True)
    source.commit('b', allow_pointless=True)
    self.build_tree(['new/'])
    made_control = self.bzrdir_format.initialize('new')
    source.branch.repository.clone(made_control)
    source.branch.clone(made_control)
    MutableTree.hooks.install_named_hook('post_build_tree', track_post_build_tree, 'Test')
    made_tree = self.workingtree_format.initialize(made_control, revision_id=a)
    self.assertEqual([a], calls)