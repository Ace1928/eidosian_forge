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
def _apply_removals(self, mover):
    """Perform tree operations that remove directory/inventory names.

        That is, delete files that are to be deleted, and put any files that
        need renaming into limbo.  This must be done in strict child-to-parent
        order.

        If inventory_delta is None, no inventory delta generation is performed.
        """
    tree_paths = sorted(self._tree_path_ids.items(), reverse=True)
    with ui.ui_factory.nested_progress_bar() as child_pb:
        for num, (path, trans_id) in enumerate(tree_paths):
            if path == '':
                continue
            child_pb.update(gettext('removing file'), num, len(tree_paths))
            full_path = self._tree.abspath(path)
            if trans_id in self._removed_contents:
                delete_path = os.path.join(self._deletiondir, trans_id)
                mover.pre_delete(full_path, delete_path)
            elif trans_id in self._new_name or trans_id in self._new_parent:
                try:
                    mover.rename(full_path, self._limbo_name(trans_id))
                except TransformRenameFailed as e:
                    if e.errno != errno.ENOENT:
                        raise
                else:
                    self.rename_count += 1