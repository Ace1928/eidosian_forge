import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_show_branch_change_no_new(self):
    tree = self.setup_ab_tree()
    tree.branch.set_last_revision_info(2, b'2b')
    s = StringIO()
    log.show_branch_change(tree.branch, s, 3, b'3b')
    self.assertContainsRe(s.getvalue(), 'Removed Revisions:')
    self.assertNotContainsRe(s.getvalue(), 'Added Revisions:')