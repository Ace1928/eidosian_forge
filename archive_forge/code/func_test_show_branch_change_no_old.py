import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_show_branch_change_no_old(self):
    tree = self.setup_ab_tree()
    s = StringIO()
    log.show_branch_change(tree.branch, s, 2, b'2b')
    self.assertContainsRe(s.getvalue(), 'Added Revisions:')
    self.assertNotContainsRe(s.getvalue(), 'Removed Revisions:')