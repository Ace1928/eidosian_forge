import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_range_open_end(self):
    self.make_file('hello', '42')
    self.run_bzr('commit -m that')
    os.chdir('../feature')
    self.make_file('hoi', 'my data')
    self.run_bzr('add')
    self.run_bzr('commit -m this')
    self.make_file('hooi', 'your data')
    self.run_bzr('add')
    self.run_bzr('commit -m that')
    self.make_file('hoooi', "someone else's data")
    self.run_bzr('add')
    self.run_bzr('commit -m these')
    self.assertEqual('', self.run_bzr('rebase -r4.. ../main')[0])
    self.assertEqual('3\n', self.run_bzr('revno')[0])
    self.assertPathDoesNotExist('hoi')
    self.assertPathDoesNotExist('hooi')
    branch = Branch.open('.')
    self.assertEqual('these', branch.repository.get_revision(branch.last_revision()).message)
    self.assertPathExists('hoooi')