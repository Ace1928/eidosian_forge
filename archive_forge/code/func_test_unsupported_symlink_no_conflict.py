import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_unsupported_symlink_no_conflict(self):

    def tt_helper():
        wt = self.make_branch_and_tree('.')
        tt = wt.transform()
        self.addCleanup(tt.finalize)
        tt.new_symlink('foo', tt.root, 'bar')
        result = tt.find_raw_conflicts()
        self.assertEqual([], result)
    os_symlink = getattr(os, 'symlink', None)
    os.symlink = None
    try:
        tt_helper()
    finally:
        if os_symlink:
            os.symlink = os_symlink