import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_quiet(self):
    t = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    t.add('file')
    t.commit('commit 1')
    self.run_bzr('push -d tree pushed-to')
    push_loc = t.branch.controldir.open_branch().get_push_location()
    out, err = self.run_bzr('push', working_dir='tree')
    self.assertEqual('Using saved push location: %s\nNo new revisions or tags to push.\n' % urlutils.local_path_from_url(push_loc), err)
    out, err = self.run_bzr('push -q', working_dir='tree')
    self.assertEqual('', out)
    self.assertEqual('', err)