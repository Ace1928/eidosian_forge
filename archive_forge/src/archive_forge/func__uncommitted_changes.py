import re
from breezy import (branch, controldir, directory_service, errors, osutils,
from breezy.bzr import bzrdir, knitrepo
from breezy.tests import http_server, scenarios, script, test_foreign
from breezy.transport import memory
def _uncommitted_changes(self):
    self.make_local_branch_and_tree()
    self.build_tree_contents([('local/file', b'in progress')])