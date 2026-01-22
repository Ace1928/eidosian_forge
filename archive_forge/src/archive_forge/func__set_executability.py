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
def _set_executability(self, path, trans_id):
    """Set the executability of versioned files """
    if self._tree._supports_executable():
        new_executability = self._new_executability[trans_id]
        abspath = self._tree.abspath(path)
        current_mode = os.stat(abspath).st_mode
        if new_executability:
            umask = os.umask(0)
            os.umask(umask)
            to_mode = current_mode | 64 & ~umask
            if current_mode & 4:
                to_mode |= 1 & ~umask
            if current_mode & 32:
                to_mode |= 8 & ~umask
        else:
            to_mode = current_mode & ~73
        osutils.chmod_if_possible(abspath, to_mode)