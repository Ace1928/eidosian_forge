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
def test_reference_parent(self):
    tree = self.make_branch_and_tree('tree')
    subtree = self.make_branch_and_tree('tree/subtree')
    subtree.commit('a change')
    try:
        tree.add_reference(subtree)
    except errors.UnsupportedOperation:
        raise tests.TestNotApplicable('Tree cannot hold references.')
    if not getattr(tree.branch._format, 'supports_reference_locations', False):
        raise tests.TestNotApplicable('Branch cannot hold reference locations.')
    tree.commit('Add reference.')
    reference_parent = tree.reference_parent(urlutils.relative_url(urlutils.strip_segment_parameters(tree.branch.user_url), urlutils.strip_segment_parameters(subtree.branch.user_url)))
    self.assertEqual(subtree.branch.user_url, reference_parent.user_url)