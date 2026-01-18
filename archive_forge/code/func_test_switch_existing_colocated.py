import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_existing_colocated(self):
    repo = self.make_repository('branch-1', format='development-colo')
    target_branch = repo.controldir.create_branch(name='foo')
    repo.controldir.set_branch_reference(target_branch)
    tree = repo.controldir.create_workingtree()
    self.build_tree(['branch-1/file-1', 'branch-1/file-2'])
    tree.add('file-1')
    revid1 = tree.commit('rev1')
    tree.add('file-2')
    revid2 = tree.commit('rev2')
    otherbranch = tree.controldir.create_branch(name='anotherbranch')
    otherbranch.generate_revision_history(revid1)
    self.run_bzr(['switch', 'anotherbranch'], working_dir='branch-1')
    tree = WorkingTree.open('branch-1')
    self.assertEqual(tree.last_revision(), revid1)
    self.assertEqual(tree.branch.control_url, otherbranch.control_url)