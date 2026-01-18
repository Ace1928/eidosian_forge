import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_verbose_no_files(self):
    """Pull --verbose should not list modified files"""
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree(['tree_a/foo'])
    tree_a.add('foo')
    tree_a.commit('bar')
    tree_b = self.make_branch_and_tree('tree_b')
    out = self.run_bzr('pull --verbose -d tree_b tree_a')[0]
    self.assertContainsRe(out, 'bar')
    self.assertNotContainsRe(out, 'added:')
    self.assertNotContainsRe(out, 'foo')