from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_simple_binding(self):
    tree = self.make_branch_and_tree('base')
    self.build_tree(['base/a', 'base/b'])
    tree.add('a', ids=b'b')
    tree.commit(message='init')
    tree.controldir.sprout('child')
    self.run_bzr('bind ../base', working_dir='child')
    d = controldir.ControlDir.open('child')
    self.assertNotEqual(None, d.open_branch().get_master_branch())
    self.run_bzr('unbind', working_dir='child')
    self.assertEqual(None, d.open_branch().get_master_branch())
    self.run_bzr('unbind', retcode=3, working_dir='child')