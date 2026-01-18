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
def test_set_get_reference_info(self):
    tree = self.make_tree_with_reference('branch', 'path/to/location')
    tree = WorkingTree.open('branch')
    branch_location = tree.get_reference_info('path/to/file')
    self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), 'path/to/location'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), branch_location))