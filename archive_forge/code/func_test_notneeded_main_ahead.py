import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_notneeded_main_ahead(self):
    self.make_file('barbla', 'bloe')
    self.run_bzr('add')
    self.run_bzr('commit -m bloe')
    os.chdir('../feature')
    self.assertEqual('Base branch is descendant of current branch. Pulling instead.\n', self.run_bzr('rebase ../main')[0])
    self.assertEqual(Branch.open('../feature').last_revision_info(), Branch.open('../main').last_revision_info())