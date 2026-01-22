import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def _create_sample_tree(self):
    tree = self.make_branch_and_tree('branch-1')
    self.build_tree(['branch-1/file-1', 'branch-1/file-2'])
    tree.add('file-1')
    tree.commit('rev1')
    tree.add('file-2')
    tree.commit('rev2')
    return tree