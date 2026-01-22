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
def _make_inv_entries(self, ordered_entries, specific_files=None):
    for trans_id, parent_file_id in ordered_entries:
        file_id = self._transform.final_file_id(trans_id)
        if file_id is None:
            continue
        if specific_files is not None and self._final_paths.get_path(trans_id) not in specific_files:
            continue
        kind = self._transform.final_kind(trans_id)
        if kind is None:
            kind = self._transform._tree.stored_kind(self._transform._tree.id2path(file_id))
        new_entry = inventory.make_entry(kind, self._transform.final_name(trans_id), parent_file_id, file_id)
        yield (new_entry, trans_id)