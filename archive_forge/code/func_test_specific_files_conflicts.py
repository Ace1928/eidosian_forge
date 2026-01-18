import codecs
import sys
from io import BytesIO, StringIO
from os import chdir, mkdir, rmdir, unlink
import breezy.branch
from breezy.bzr import bzrdir, conflicts
from ... import errors, osutils, status
from ...osutils import pathjoin
from ...revisionspec import RevisionSpec
from ...status import show_tree_status
from ...workingtree import WorkingTree
from .. import TestCaseWithTransport, TestSkipped
def test_specific_files_conflicts(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir2/'])
    tree.add('dir2')
    tree.commit('added dir2')
    tree.set_conflicts([conflicts.ContentsConflict('foo')])
    tof = BytesIO()
    show_tree_status(tree, specific_files=['dir2'], to_file=tof)
    self.assertEqualDiff(b'', tof.getvalue())
    tree.set_conflicts([conflicts.ContentsConflict('dir2')])
    tof = StringIO()
    show_tree_status(tree, specific_files=['dir2'], to_file=tof)
    self.assertEqualDiff('conflicts:\n  Contents conflict in dir2\n', tof.getvalue())
    tree.set_conflicts([conflicts.ContentsConflict('dir2/file1')])
    tof = StringIO()
    show_tree_status(tree, specific_files=['dir2'], to_file=tof)
    self.assertEqualDiff('conflicts:\n  Contents conflict in dir2/file1\n', tof.getvalue())