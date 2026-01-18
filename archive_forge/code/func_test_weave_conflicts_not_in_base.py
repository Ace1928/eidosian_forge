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
def test_weave_conflicts_not_in_base(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    a_id = builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', None))])
    b_id = builder.build_snapshot([a_id], [])
    c_id = builder.build_snapshot([a_id], [('add', ('foo', b'foo-id', 'file', b'orig\ncontents\n'))])
    d_id = builder.build_snapshot([b_id, c_id], [('add', ('foo', b'foo-id', 'file', b'orig\ncontents\nand D\n'))])
    e_id = builder.build_snapshot([c_id, b_id], [('modify', ('foo', b'orig\ncontents\nand E\n'))])
    builder.finish_series()
    tree = builder.get_branch().create_checkout('tree', lightweight=True)
    self.assertEqual(1, len(tree.merge_from_branch(tree.branch, to_revision=d_id, merge_type=WeaveMerger)))
    self.assertPathExists('tree/foo.THIS')
    self.assertPathExists('tree/foo.OTHER')
    self.expectFailure('fail to create .BASE in some criss-cross merges', self.assertPathExists, 'tree/foo.BASE')
    self.assertPathExists('tree/foo.BASE')