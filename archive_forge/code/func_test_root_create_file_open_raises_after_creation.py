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
def test_root_create_file_open_raises_after_creation(self):
    tt, trans_id = self.create_transform_and_root_trans_id()
    self._override_globals_in_method(tt, 'create_file', {'open': self._fake_open_raises_after})
    self.assertRaises(RuntimeError, tt.create_file, [b'contents'], trans_id)
    path = tt._limbo_name(trans_id)
    self.assertPathExists(path)
    tt.finalize()
    self.assertPathDoesNotExist(path)
    self.assertPathDoesNotExist(tt._limbodir)