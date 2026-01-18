import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
def test_symlink_merge(self):
    if sys.platform != 'win32':
        builder = MergeBuilder(getcwd())
        name1 = builder.add_symlink(builder.root(), 'name1', 'target1', file_id=b'1')
        name2 = builder.add_symlink(builder.root(), 'name2', 'target1', file_id=b'2')
        name3 = builder.add_symlink(builder.root(), 'name3', 'target1', file_id=b'3')
        builder.change_target(name1, this=b'target2')
        builder.change_target(name2, base=b'target2')
        builder.change_target(name3, other=b'target2')
        builder.merge()
        self.assertEqual(builder.this.get_symlink_target('name1'), 'target2')
        self.assertEqual(builder.this.get_symlink_target('name2'), 'target1')
        self.assertEqual(builder.this.get_symlink_target('name3'), 'target2')
        builder.cleanup()