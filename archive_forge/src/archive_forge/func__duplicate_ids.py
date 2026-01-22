import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def _duplicate_ids(self):
    """Each inventory id may only be used once"""
    conflicts = []
    try:
        all_ids = self._tree.all_file_ids()
    except errors.UnsupportedOperation:
        return []
    removed_tree_ids = {self.tree_file_id(trans_id) for trans_id in self._removed_id}
    active_tree_ids = all_ids.difference(removed_tree_ids)
    for trans_id, file_id in self._new_id.items():
        if file_id in active_tree_ids:
            path = self._tree.id2path(file_id)
            old_trans_id = self.trans_id_tree_path(path)
            conflicts.append(('duplicate id', old_trans_id, trans_id))
    return conflicts