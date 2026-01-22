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
def _get_potential_orphans(self, dir_id):
    """Find the potential orphans in a directory.

        A directory can't be safely deleted if there are versioned files in it.
        If all the contained files are unversioned then they can be orphaned.

        The 'None' return value means that the directory contains at least one
        versioned file and should not be deleted.

        :param dir_id: The directory trans id.

        :return: A list of the orphan trans ids or None if at least one
             versioned file is present.
        """
    orphans = []
    for child_tid in self.by_parent()[dir_id]:
        if child_tid in self._removed_contents:
            continue
        if not self.final_is_versioned(child_tid):
            orphans.append(child_tid)
        else:
            orphans = None
            break
    return orphans