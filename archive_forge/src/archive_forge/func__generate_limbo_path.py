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
def _generate_limbo_path(self, trans_id):
    """Generate a limbo path using the final path if possible.

        This optimizes the performance of applying the tree transform by
        avoiding renames.  These renames can be avoided only when the parent
        directory is already scheduled for creation.

        If the final path cannot be used, falls back to using the trans_id as
        the relpath.
        """
    parent = self._new_parent.get(trans_id)
    use_direct_path = False
    if self._new_contents.get(parent) == 'directory':
        filename = self._new_name.get(trans_id)
        if filename is not None:
            if parent not in self._limbo_children:
                self._limbo_children[parent] = set()
                self._limbo_children_names[parent] = {}
                use_direct_path = True
            elif self._case_sensitive_target:
                if self._limbo_children_names[parent].get(filename) in (trans_id, None):
                    use_direct_path = True
            else:
                for l_filename, l_trans_id in self._limbo_children_names[parent].items():
                    if l_trans_id == trans_id:
                        continue
                    if l_filename.lower() == filename.lower():
                        break
                else:
                    use_direct_path = True
    if not use_direct_path:
        return DiskTreeTransform._generate_limbo_path(self, trans_id)
    limbo_name = osutils.pathjoin(self._limbo_files[parent], filename)
    self._limbo_children[parent].add(trans_id)
    self._limbo_children_names[parent][filename] = trans_id
    return limbo_name