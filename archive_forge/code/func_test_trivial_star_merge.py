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
def test_trivial_star_merge(self):
    """Test that merges in a star shape Just Work."""
    self.build_tree(('original/', 'original/file1', 'original/file2'))
    tree = self.make_branch_and_tree('original')
    branch = tree.branch
    tree.smart_add(['original'])
    tree.commit('start branch.', verbose=False)
    self.build_tree(('mary/',))
    branch.controldir.clone('mary')
    with open('original/file1', 'w') as f:
        f.write('John\n')
    tree.commit('change file1')
    mary_tree = WorkingTree.open('mary')
    mary_branch = mary_tree.branch
    with open('mary/file2', 'w') as f:
        f.write('Mary\n')
    mary_tree.commit('change file2')
    base = [None, None]
    other = ('mary', -1)
    tree.merge_from_branch(mary_tree.branch)
    with open('original/file1') as f:
        self.assertEqual('John\n', f.read())
    with open('original/file2') as f:
        self.assertEqual('Mary\n', f.read())