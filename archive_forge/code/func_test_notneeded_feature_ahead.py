import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_notneeded_feature_ahead(self):
    os.chdir('../feature')
    self.make_file('barbla', 'bloe')
    self.run_bzr('add')
    self.run_bzr('commit -m bloe')
    self.assertEqual('No revisions to rebase.\n', self.run_bzr('rebase ../main')[0])