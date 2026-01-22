import errno
import os
import posixpath
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from dulwich.index import blob_from_path_and_stat, commit_tree
from dulwich.objects import Blob
from .. import annotate, conflicts, errors, multiparent, osutils
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import ui, urlutils
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import InterTree, TreeChange
from .mapping import (decode_git_path, encode_git_path, mode_is_executable,
from .tree import GitTree, GitTreeDirectory, GitTreeFile, GitTreeSymlink
def _affected_ids(self):
    """Return the set of transform ids affected by the transform"""
    trans_ids = set(self._removed_id)
    trans_ids.update(self._versioned)
    trans_ids.update(self._removed_contents)
    trans_ids.update(self._new_contents)
    trans_ids.update(self._new_executability)
    trans_ids.update(self._new_name)
    trans_ids.update(self._new_parent)
    return trans_ids