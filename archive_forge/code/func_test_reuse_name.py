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
def test_reuse_name(self):
    """Avoid reusing the same limbo name for different files"""
    transform, root = self.transform()
    parent = transform.new_directory('parent', root)
    transform.new_directory('child', parent)
    try:
        child2 = transform.new_directory('child', parent)
    except OSError:
        self.fail('Tranform tried to use the same limbo name twice')
    transform.adjust_path('child2', parent, child2)
    transform.apply()
    self.assertEqual(2, transform.rename_count)