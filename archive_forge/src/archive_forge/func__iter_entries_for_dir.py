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
def _iter_entries_for_dir(self, dir_path):
    """Return path, entry for items in a directory without recursing down."""
    ordered_ids = []
    dir_trans_id = self._path2trans_id(dir_path)
    dir_id = self._transform.final_file_id(dir_trans_id)
    for child_trans_id in self._all_children(dir_trans_id):
        ordered_ids.append((child_trans_id, dir_id))
    path_entries = []
    for entry, trans_id in self._make_inv_entries(ordered_ids):
        path_entries.append((self._final_paths.get_path(trans_id), entry))
    path_entries.sort()
    return path_entries