import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_remember(self):
    """Push changes from one branch to another and test push location."""
    transport = self.get_transport()
    tree_a = self.make_branch_and_tree('branch_a')
    branch_a = tree_a.branch
    self.build_tree(['branch_a/a'])
    tree_a.add('a')
    tree_a.commit('commit a')
    tree_b = branch_a.controldir.sprout('branch_b').open_workingtree()
    branch_b = tree_b.branch
    tree_c = branch_a.controldir.sprout('branch_c').open_workingtree()
    branch_c = tree_c.branch
    self.build_tree(['branch_a/b'])
    tree_a.add('b')
    tree_a.commit('commit b')
    self.build_tree(['branch_b/c'])
    tree_b.add('c')
    tree_b.commit('commit c')
    self.assertEqual(None, branch_b.get_push_location())
    out = self.run_bzr('push', working_dir='branch_a', retcode=3)
    self.assertEqual(out, ('', 'brz: ERROR: No push location known or specified.\n'))
    self.run_bzr('push path/which/doesnt/exist', working_dir='branch_a', retcode=3)
    out = self.run_bzr('push', working_dir='branch_a', retcode=3)
    self.assertEqual(('', 'brz: ERROR: No push location known or specified.\n'), out)
    out = self.run_bzr('push ../branch_b', working_dir='branch_a', retcode=3)
    self.assertEqual(out, ('', 'brz: ERROR: These branches have diverged.  See "brz help diverged-branches" for more information.\n'))
    branch_a = branch_a.controldir.open_branch()
    self.assertEqual(osutils.abspath(branch_a.get_push_location()), osutils.abspath(branch_b.controldir.root_transport.base))
    uncommit.uncommit(branch=branch_b, tree=tree_b)
    transport.delete('branch_b/c')
    out, err = self.run_bzr('push', working_dir='branch_a')
    branch_a = branch_a.controldir.open_branch()
    path = branch_a.get_push_location()
    self.assertEqual(err, 'Using saved push location: %s\nAll changes applied successfully.\nPushed up to revision 2.\n' % urlutils.local_path_from_url(path))
    self.assertEqual(path, branch_b.controldir.root_transport.base)
    self.run_bzr('push ../branch_c --remember', working_dir='branch_a')
    branch_a = branch_a.controldir.open_branch()
    self.assertEqual(branch_a.get_push_location(), branch_c.controldir.root_transport.base)