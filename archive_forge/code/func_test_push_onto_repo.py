import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_onto_repo(self):
    """We should be able to 'brz push' into an existing bzrdir."""
    tree = self.create_simple_tree()
    repo = self.make_repository('repo', shared=True)
    self.run_bzr('push ../repo', working_dir='tree')
    self.assertRaises(errors.NoWorkingTree, workingtree.WorkingTree.open, 'repo')
    new_branch = branch.Branch.open('repo')
    self.assertEqual(tree.last_revision(), new_branch.last_revision())