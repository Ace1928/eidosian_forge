import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_save_false(self):
    """Dry-run add doesn't permanently affect the tree."""
    wt = self.make_branch_and_tree('.')
    with wt.lock_write():
        self.build_tree(['file'])
        wt.smart_add(['file'], save=False)
        self.assertFalse(wt.is_versioned('file'))
    wt = wt.controldir.open_workingtree()
    self.assertFalse(wt.is_versioned('file'))