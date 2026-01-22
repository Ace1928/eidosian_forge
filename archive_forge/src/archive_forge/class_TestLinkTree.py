import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
class TestLinkTree(tests.TestCaseWithTransport):

    def setUp(self):
        tests.TestCaseWithTransport.setUp(self)
        self.requireFeature(HardlinkFeature(self.test_dir))
        self.parent_tree = self.make_branch_and_tree('parent')
        self.parent_tree.lock_write()
        self.addCleanup(self.parent_tree.unlock)
        self.build_tree_contents([('parent/foo', b'bar')])
        self.parent_tree.add('foo')
        self.parent_tree.commit('added foo')
        child_controldir = self.parent_tree.controldir.sprout('child')
        self.child_tree = child_controldir.open_workingtree()

    def hardlinked(self):
        parent_stat = os.lstat(self.parent_tree.abspath('foo'))
        child_stat = os.lstat(self.child_tree.abspath('foo'))
        return parent_stat.st_ino == child_stat.st_ino

    def test_link_fails_if_modified(self):
        """If the file to be linked has modified text, don't link."""
        self.build_tree_contents([('child/foo', b'baz')])
        transform.link_tree(self.child_tree, self.parent_tree)
        self.assertFalse(self.hardlinked())

    def test_link_fails_if_execute_bit_changed(self):
        """If the file to be linked has modified execute bit, don't link."""
        tt = self.child_tree.transform()
        try:
            trans_id = tt.trans_id_tree_path('foo')
            tt.set_executability(True, trans_id)
            tt.apply()
        finally:
            tt.finalize()
        transform.link_tree(self.child_tree, self.parent_tree)
        self.assertFalse(self.hardlinked())

    def test_link_succeeds_if_unmodified(self):
        """If the file to be linked is unmodified, link"""
        transform.link_tree(self.child_tree, self.parent_tree)
        self.assertTrue(self.hardlinked())