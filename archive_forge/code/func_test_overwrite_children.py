import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_overwrite_children(self):
    a_tree = self.make_branch_and_tree('a')
    self.build_tree_contents([('a/foo', b'original\n')])
    a_tree.add('foo')
    a_tree.commit(message='initial commit')
    b_tree = a_tree.controldir.sprout('b').open_workingtree()
    self.build_tree_contents([('a/foo', b'changed\n')])
    a_tree.commit(message='later change')
    self.build_tree_contents([('a/foo', b'a third change')])
    a_tree.commit(message='a third change')
    self.assertEqual(a_tree.branch.last_revision_info()[0], 3)
    b_tree.merge_from_branch(a_tree.branch)
    b_tree.commit(message='merge')
    self.assertEqual(b_tree.branch.last_revision_info()[0], 2)
    self.build_tree_contents([('a/foo', b'a fourth change\n')])
    a_tree.commit(message='a fourth change')
    rev_info_a = a_tree.branch.last_revision_info()
    self.assertEqual(rev_info_a[0], 4)
    self.run_bzr('pull --overwrite ../a', working_dir='b')
    rev_info_b = b_tree.branch.last_revision_info()
    self.assertEqual(rev_info_b[0], 4)
    self.assertEqual(rev_info_b, rev_info_a)