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
class TestIllegalPaths(TestCaseWithWorkingTree):

    def test_bad_fs_path(self):
        if osutils.normalizes_filenames():
            raise tests.TestNotApplicable('OSX normalizes filenames')
        self.requireFeature(features.UTF8Filesystem)
        tree = self.make_branch_and_tree('tree')
        self.build_tree(['tree/subdir/', 'tree/subdir/somefile'])
        tree.add(['subdir', 'subdir/somefile'])
        with open(b'tree/subdir/m\xb5', 'wb') as f:
            f.write(b'trivial\n')
        tree.lock_read()
        self.addCleanup(tree.unlock)
        basis = tree.basis_tree()
        basis.lock_read()
        self.addCleanup(basis.unlock)
        changes = list(tree.iter_changes(tree.basis_tree(), want_unversioned=True))
        self.assertIn('subdir/m\udcb5', [c.path[1] for c in changes])