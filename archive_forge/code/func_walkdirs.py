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
def walkdirs(self, prefix=''):
    pending = [self._transform.root]
    while len(pending) > 0:
        parent_id = pending.pop()
        children = []
        subdirs = []
        prefix = prefix.rstrip('/')
        parent_path = self._final_paths.get_path(parent_id)
        for child_id in self._all_children(parent_id):
            path_from_root = self._final_paths.get_path(child_id)
            basename = self._transform.final_name(child_id)
            kind = self._transform.final_kind(child_id)
            if kind is not None:
                versioned_kind = kind
            else:
                kind = 'unknown'
                versioned_kind = self._transform._tree.stored_kind(path_from_root)
            if versioned_kind == 'directory':
                subdirs.append(child_id)
            children.append((path_from_root, basename, kind, None, versioned_kind))
        children.sort()
        if parent_path.startswith(prefix):
            yield (parent_path, children)
        pending.extend(sorted(subdirs, key=self._final_paths.get_path, reverse=True))