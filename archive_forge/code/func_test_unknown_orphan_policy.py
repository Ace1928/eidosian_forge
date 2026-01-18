import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def test_unknown_orphan_policy(self):
    wt = self.make_branch_and_tree('.')
    self._set_orphan_policy(wt, 'donttouchmypreciouuus')
    tt, orphan_tid = self._prepare_orphan(wt)
    warnings = []

    def warning(*args):
        warnings.append(args[0] % args[1:])
    self.overrideAttr(trace, 'warning', warning)
    remaining_conflicts = resolve_conflicts(tt)
    self.assertLength(1, remaining_conflicts)
    self.assertEqual(('deleting parent', 'Not deleting', 'new-1'), remaining_conflicts.pop())
    self.assertLength(1, warnings)
    self.assertStartsWith(warnings[0], 'Value "donttouchmypreciouuus" ')