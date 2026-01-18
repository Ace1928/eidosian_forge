import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def test_null_old(self):
    tree = self.setup_ab_tree()
    old, new = log.get_history_change(revision.NULL_REVISION, b'3a', tree.branch.repository)
    self.assertEqual([], old)
    self.assertEqual([b'1a', b'2a', b'3a'], new)