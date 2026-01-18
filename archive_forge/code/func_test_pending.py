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
def test_pending(self):
    """Pending merges display works, including Unicode"""
    mkdir('./branch')
    wt = self.make_branch_and_tree('branch')
    b = wt.branch
    wt.commit('Empty commit 1')
    b_2_dir = b.controldir.sprout('./copy')
    b_2 = b_2_dir.open_branch()
    wt2 = b_2_dir.open_workingtree()
    wt.commit('༢ Empty commit 2')
    wt2.merge_from_branch(wt.branch)
    message = self.status_string(wt2, verbose=True)
    self.assertStartsWith(message, 'pending merges:\n')
    self.assertEndsWith(message, 'Empty commit 2\n')
    wt2.commit('merged')
    wt.commit('Empty commit 3 ' + 'blah blah blah blah ' * 100)
    wt2.merge_from_branch(wt.branch)
    message = self.status_string(wt2, verbose=True)
    self.assertStartsWith(message, 'pending merges:\n')
    self.assertTrue('Empty commit 3' in message)
    self.assertEndsWith(message, '...\n')