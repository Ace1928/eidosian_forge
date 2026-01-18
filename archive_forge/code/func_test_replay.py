import os
from ....branch import Branch
from ....tests.blackbox import ExternalBase
def test_replay(self):
    os.mkdir('main')
    os.chdir('main')
    self.run_bzr('init')
    with open('bar', 'w') as f:
        f.write('42')
    self.run_bzr('add')
    self.run_bzr('commit -m that')
    os.mkdir('../feature')
    os.chdir('../feature')
    self.run_bzr('init')
    branch = Branch.open('.')
    with open('hello', 'w') as f:
        f.write('my data')
    self.run_bzr('add')
    self.run_bzr('commit -m this')
    self.assertEqual(1, branch.revno())
    self.run_bzr('replay -r1 ../main')
    self.assertEqual(2, branch.revno())
    self.assertTrue(os.path.exists('bar'))