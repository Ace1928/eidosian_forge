import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_with_revisionspec(self):
    """We should be able to push a revision older than the tip."""
    tree_from = self.make_branch_and_tree('from')
    tree_from.commit('One.', rev_id=b'from-1')
    tree_from.commit('Two.', rev_id=b'from-2')
    self.run_bzr('push -r1 ../to', working_dir='from')
    tree_to = workingtree.WorkingTree.open('to')
    repo_to = tree_to.branch.repository
    self.assertTrue(repo_to.has_revision(b'from-1'))
    self.assertFalse(repo_to.has_revision(b'from-2'))
    self.assertEqual(tree_to.branch.last_revision_info()[1], b'from-1')
    self.assertFalse(tree_to.changes_from(tree_to.basis_tree()).has_changed())
    self.run_bzr_error(['brz: ERROR: brz push --revision takes exactly one revision identifier\n'], 'push -r0..2 ../to', working_dir='from')