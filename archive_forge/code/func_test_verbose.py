import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_verbose(self):
    self.make_file('hello', '42')
    self.run_bzr('commit -m that')
    os.chdir('../feature')
    self.make_file('hoi', 'my data')
    self.run_bzr('add')
    self.run_bzr('commit -m this')
    out, err = self.run_bzr('rebase -v ../main')
    self.assertContainsRe(err, '1 revisions will be rebased:')
    self.assertEqual('', out)
    self.assertEqual('3\n', self.run_bzr('revno')[0])