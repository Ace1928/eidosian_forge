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
def _parent_type_conflicts(self, by_parent):
    """Children must have a directory parent"""
    for parent_id, children in by_parent.items():
        if parent_id == ROOT_PARENT:
            continue
        no_children = True
        for child_id in children:
            if self.final_kind(child_id) is not None:
                no_children = False
                break
        if no_children:
            continue
        kind = self.final_kind(parent_id)
        if kind is None:
            yield ('missing parent', parent_id)
        elif kind != 'directory':
            yield ('non-directory parent', parent_id)