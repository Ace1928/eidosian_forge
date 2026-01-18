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
def test_rename_in_limbo_rename_raises_before_rename(self):
    tt, trans_id = self.create_transform_and_root_trans_id()
    parent1 = tt.new_directory('parent1', tt.root)
    child1 = tt.new_file('child1', parent1, [b'contents'])
    parent2 = tt.new_directory('parent2', tt.root)

    class FakeOSModule:

        def rename(self, old, new):
            raise RuntimeError
    self._override_globals_in_method(tt, '_rename_in_limbo', {'os': FakeOSModule()})
    self.assertRaises(RuntimeError, tt.adjust_path, 'child1', parent2, child1)
    path = osutils.pathjoin(tt._limbo_name(parent1), 'child1')
    self.assertPathExists(path)
    tt.finalize()
    self.assertPathDoesNotExist(path)
    self.assertPathDoesNotExist(tt._limbodir)