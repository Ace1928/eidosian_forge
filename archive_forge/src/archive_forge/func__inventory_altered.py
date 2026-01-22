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
def _inventory_altered(self):
    """Determine which trans_ids need new Inventory entries.

        An new entry is needed when anything that would be reflected by an
        inventory entry changes, including file name, file_id, parent file_id,
        file kind, and the execute bit.

        Some care is taken to return entries with real changes, not cases
        where the value is deleted and then restored to its original value,
        but some actually unchanged values may be returned.

        :returns: A list of (path, trans_id) for all items requiring an
            inventory change. Ordered by path.
        """
    changed_ids = set()
    new_file_id = {t for t in self._new_id if self._new_id[t] != self.tree_file_id(t)}
    for id_set in [self._new_name, self._new_parent, new_file_id, self._new_executability]:
        changed_ids.update(id_set)
    changed_kind = set(self._removed_contents)
    changed_kind.intersection_update(self._new_contents)
    changed_kind.difference_update(changed_ids)
    changed_kind = (t for t in changed_kind if self.tree_kind(t) != self.final_kind(t))
    changed_ids.update(changed_kind)
    for parent_trans_id in new_file_id:
        changed_ids.update(self.iter_tree_children(parent_trans_id))
    return sorted(FinalPaths(self).get_paths(changed_ids))