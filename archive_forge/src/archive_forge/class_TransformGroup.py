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
class TransformGroup:

    def __init__(self, dirname, root_id):
        self.name = dirname
        os.mkdir(dirname)
        self.wt = ControlDir.create_standalone_workingtree(dirname)
        if self.wt.supports_file_ids:
            self.wt.set_root_id(root_id)
        self.b = self.wt.branch
        self.tt = self.wt.transform()
        self.root = self.tt.trans_id_tree_path('')