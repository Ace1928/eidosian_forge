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
def test_perms_merge(self):
    builder = MergeBuilder(getcwd())
    name1 = builder.add_file(builder.root(), 'name1', b'text1', True, file_id=b'1')
    builder.change_perms(name1, other=False)
    name2 = builder.add_file(builder.root(), 'name2', b'text2', True, file_id=b'2')
    builder.change_perms(name2, base=False)
    name3 = builder.add_file(builder.root(), 'name3', b'text3', True, file_id=b'3')
    builder.change_perms(name3, this=False)
    name4 = builder.add_file(builder.root(), 'name4', b'text4', False, file_id=b'4')
    builder.change_perms(name4, this=True)
    builder.remove_file(name4, base=True)
    builder.merge()
    self.assertIs(builder.this.is_executable('name1'), False)
    self.assertIs(builder.this.is_executable('name2'), True)
    self.assertIs(builder.this.is_executable('name3'), False)
    builder.cleanup()