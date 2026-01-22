import contextlib
import errno
import os
import time
from stat import S_IEXEC, S_ISREG
from typing import Callable
from . import config as _mod_config
from . import controldir, errors, lazy_import, lock, osutils, registry, trace
from breezy import (
from breezy.i18n import gettext
from .errors import BzrError, DuplicateKey, InternalBzrError
from .filters import ContentFilterContext, filtered_output_bytes
from .mutabletree import MutableTree
from .osutils import delete_any, file_kind, pathjoin, sha_file, splitpath
from .progress import ProgressPhase
from .transport import FileExists, NoSuchFile
from .tree import InterTree, find_previous_path
class FinalPaths:
    """Make path calculation cheap by memoizing paths.

    The underlying tree must not be manipulated between calls, or else
    the results will likely be incorrect.
    """

    def __init__(self, transform):
        object.__init__(self)
        self._known_paths = {}
        self.transform = transform

    def _determine_path(self, trans_id):
        if trans_id == self.transform.root or trans_id == ROOT_PARENT:
            return ''
        name = self.transform.final_name(trans_id)
        parent_id = self.transform.final_parent(trans_id)
        if parent_id == self.transform.root:
            return name
        else:
            return pathjoin(self.get_path(parent_id), name)

    def get_path(self, trans_id):
        """Find the final path associated with a trans_id"""
        if trans_id not in self._known_paths:
            self._known_paths[trans_id] = self._determine_path(trans_id)
        return self._known_paths[trans_id]

    def get_paths(self, trans_ids):
        return [(self.get_path(t), t) for t in trans_ids]