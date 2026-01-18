import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def test_push_funky_id(self):
    t = self.make_branch_and_tree('tree')
    self.build_tree(['tree/filename'])
    t.add('filename', ids=b'funky-chars<>%&;"\'')
    t.commit('commit filename')
    self.run_bzr('push -d tree new-tree')