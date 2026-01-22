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
class TestControlComponent(TestCaseWithWorkingTree):
    """WorkingTree implementations adequately implement ControlComponent."""

    def test_urls(self):
        wt = self.make_branch_and_tree('wt')
        self.assertIsInstance(wt.user_url, str)
        self.assertEqual(wt.user_url, wt.user_transport.base)
        self.assertEqual(wt.control_url, wt.control_transport.base)