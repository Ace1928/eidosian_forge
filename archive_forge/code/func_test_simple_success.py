import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_simple_success(self):
    self.make_file('hello', '42')
    self.run_bzr('commit -m that')
    os.chdir('../feature')
    self.make_file('hoi', 'my data')
    self.run_bzr('add')
    self.run_bzr('commit -m this')
    self.assertEqual('', self.run_bzr('rebase ../main')[0])
    self.assertEqual('3\n', self.run_bzr('revno')[0])