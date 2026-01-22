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
def _add_tree_children(self):
    """Add all the children of all active parents to the known paths.

        Active parents are those which gain children, and those which are
        removed.  This is a necessary first step in detecting conflicts.
        """
    parents = list(self.by_parent())
    parents.extend([t for t in self._removed_contents if self.tree_kind(t) == 'directory'])
    for trans_id in self._removed_id:
        path = self.tree_path(trans_id)
        if path is not None:
            try:
                if self._tree.stored_kind(path) == 'directory':
                    parents.append(trans_id)
            except _mod_transport.NoSuchFile:
                pass
        elif self.tree_kind(trans_id) == 'directory':
            parents.append(trans_id)
    for parent_id in parents:
        list(self.iter_tree_children(parent_id))