import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_show_branch_change(self):
    tree = self.setup_ab_tree()
    s = StringIO()
    log.show_branch_change(tree.branch, s, 3, b'3a')
    self.assertContainsRe(s.getvalue(), '[*]{60}\nRemoved Revisions:\n(.|\n)*2a(.|\n)*3a(.|\n)*[*]{60}\n\nAdded Revisions:\n(.|\n)*2b(.|\n)*3b')