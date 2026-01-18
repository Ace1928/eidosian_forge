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
def test_status_versioned(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['hello.txt'])
    result = self.run_bzr('status --versioned')[0]
    self.assertNotContainsRe(result, 'unknown:\n  hello.txt\n')
    tree.add('hello.txt')
    result = self.run_bzr('status --versioned')[0]
    self.assertContainsRe(result, 'added:\n  hello.txt\n')
    tree.commit('added')
    result = self.run_bzr('status --versioned -r 0..1')[0]
    self.assertContainsRe(result, 'added:\n  hello.txt\n')
    self.build_tree(['world.txt'])
    result = self.run_bzr('status --versioned -r 0')[0]
    self.assertContainsRe(result, 'added:\n  hello.txt\n')
    self.assertNotContainsRe(result, 'unknown:\n  world.txt\n')
    result2 = self.run_bzr('status --versioned -r 0..')[0]
    self.assertEqual(result2, result)