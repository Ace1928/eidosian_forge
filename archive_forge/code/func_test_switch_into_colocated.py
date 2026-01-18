import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_into_colocated(self):
    tree = self.make_branch_and_tree('.', format='development-colo')
    self.build_tree(['file-1', 'file-2'])
    tree.add('file-1')
    revid1 = tree.commit('rev1')
    tree.add('file-2')
    revid2 = tree.commit('rev2')
    self.run_bzr(['switch', '-b', 'anotherbranch'])
    self.assertEqual({'', 'anotherbranch'}, set(tree.branch.controldir.branch_names()))