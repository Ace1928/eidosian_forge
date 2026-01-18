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