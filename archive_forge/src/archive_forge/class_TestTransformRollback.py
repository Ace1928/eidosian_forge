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
class TestTransformRollback(tests.TestCaseWithTransport):

    class ExceptionFileMover(_FileMover):

        def __init__(self, bad_source=None, bad_target=None):
            _FileMover.__init__(self)
            self.bad_source = bad_source
            self.bad_target = bad_target

        def rename(self, source, target):
            if self.bad_source is not None and source.endswith(self.bad_source):
                raise Bogus
            elif self.bad_target is not None and target.endswith(self.bad_target):
                raise Bogus
            else:
                _FileMover.rename(self, source, target)

    def test_rollback_rename(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b'])
        tt = tree.transform()
        self.addCleanup(tt.finalize)
        a_id = tt.trans_id_tree_path('a')
        tt.adjust_path('c', tt.root, a_id)
        tt.adjust_path('d', a_id, tt.trans_id_tree_path('a/b'))
        self.assertRaises(Bogus, tt.apply, _mover=self.ExceptionFileMover(bad_source='a'))
        self.assertPathExists('a')
        self.assertPathExists('a/b')
        tt.apply()
        self.assertPathExists('c')
        self.assertPathExists('c/d')

    def test_rollback_rename_into_place(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b'])
        tt = tree.transform()
        self.addCleanup(tt.finalize)
        a_id = tt.trans_id_tree_path('a')
        tt.adjust_path('c', tt.root, a_id)
        tt.adjust_path('d', a_id, tt.trans_id_tree_path('a/b'))
        self.assertRaises(Bogus, tt.apply, _mover=self.ExceptionFileMover(bad_target='c/d'))
        self.assertPathExists('a')
        self.assertPathExists('a/b')
        tt.apply()
        self.assertPathExists('c')
        self.assertPathExists('c/d')

    def test_rollback_deletion(self):
        tree = self.make_branch_and_tree('.')
        self.build_tree(['a/', 'a/b'])
        tt = tree.transform()
        self.addCleanup(tt.finalize)
        a_id = tt.trans_id_tree_path('a')
        tt.delete_contents(a_id)
        tt.adjust_path('d', tt.root, tt.trans_id_tree_path('a/b'))
        self.assertRaises(Bogus, tt.apply, _mover=self.ExceptionFileMover(bad_target='d'))
        self.assertPathExists('a')
        self.assertPathExists('a/b')