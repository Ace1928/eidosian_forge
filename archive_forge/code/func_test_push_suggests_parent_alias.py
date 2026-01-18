import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_suggests_parent_alias(self):
    """Push suggests using :parent if there is a known parent branch."""
    tree_a = self.make_branch_and_tree('a')
    tree_a.commit('this is a commit')
    tree_b = self.make_branch_and_tree('b')
    out = self.run_bzr('push', working_dir='a', retcode=3)
    self.assertEqual(out, ('', 'brz: ERROR: No push location known or specified.\n'))
    tree_a.branch.set_parent(tree_b.branch.base)
    out = self.run_bzr('push', working_dir='a', retcode=3)
    self.assertEqual(out, ('', "brz: ERROR: No push location known or specified. To push to the parent branch (at %s), use 'brz push :parent'.\n" % urlutils.unescape_for_display(tree_b.branch.base, 'utf-8')))