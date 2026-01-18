from breezy import branch, controldir, errors, tests
from breezy.tests import script
def test_pull_updates_both(self):
    base_tree = self.create_branches()[0]
    newchild_tree = base_tree.controldir.sprout('newchild').open_workingtree()
    self.build_tree_contents([('newchild/b', b'newchild b contents\n')])
    newchild_tree.commit(message='newchild')
    self.check_revno(2, 'newchild')
    self.run_bzr('pull ../newchild', working_dir='child')
    self.check_revno(2, 'child')
    self.check_revno(2, 'base')