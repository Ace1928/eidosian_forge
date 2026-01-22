import os
import breezy.osutils
from breezy.tests import TestCaseWithTransport
from breezy.trace import mutter
from breezy.workingtree import WorkingTree
def _prepare_rename_mod_tree(self):
    self.build_tree(['a/', 'a/b', 'a/c', 'a/d/', 'a/d/e', 'f/', 'f/g', 'f/h', 'f/i'])
    self.run_bzr('init')
    self.run_bzr('add')
    self.run_bzr('commit -m 1')
    wt = WorkingTree.open('.')
    wt.rename_one('a/b', 'f/b')
    wt.rename_one('a/d/e', 'f/e')
    wt.rename_one('a/d', 'f/d')
    wt.rename_one('f/g', 'a/g')
    wt.rename_one('f/h', 'h')
    wt.rename_one('f', 'j')