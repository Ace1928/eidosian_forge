import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_new_branch_revision_count(self):
    t = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    t.add('file')
    t.commit('commit 1')
    out, err = self.run_bzr('push -d tree pushed-to')
    self.assertEqual('', out)
    self.assertEqual('Created new branch.\n', err)