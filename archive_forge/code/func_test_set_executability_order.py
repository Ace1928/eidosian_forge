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
def test_set_executability_order(self):
    """Ensure that executability behaves the same, no matter what order.

        - create file and set executability simultaneously
        - create file and set executability afterward
        - unsetting the executability of a file whose executability has not
          been
        declared should throw an exception (this may happen when a
        merge attempts to create a file with a duplicate ID)
        """
    transform, root = self.transform()
    wt = transform._tree
    wt.lock_read()
    self.addCleanup(wt.unlock)
    transform.new_file('set_on_creation', root, [b'Set on creation'], b'soc', True)
    sac = transform.new_file('set_after_creation', root, [b'Set after creation'], b'sac')
    transform.set_executability(True, sac)
    uws = transform.new_file('unset_without_set', root, [b'Unset badly'], b'uws')
    self.assertRaises(KeyError, transform.set_executability, None, uws)
    transform.apply()
    self.assertTrue(wt.is_executable('set_on_creation'))
    self.assertTrue(wt.is_executable('set_after_creation'))