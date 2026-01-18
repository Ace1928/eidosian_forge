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
def test_update_references_retains_old_references(self):
    tree = self.make_tree_with_reference('branch', 'reference')
    new_tree = self.make_tree_with_reference('new_branch', 'reference2')
    new_tree.branch.update_references(tree.branch)
    self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('path/to/file')))