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
def _get_file_id_maps(self):
    """Return mapping of file_ids to trans_ids in the to and from states"""
    trans_ids = self._affected_ids()
    from_trans_ids = {}
    to_trans_ids = {}
    for trans_id in trans_ids:
        from_file_id = self.tree_file_id(trans_id)
        if from_file_id is not None:
            from_trans_ids[from_file_id] = trans_id
        to_file_id = self.final_file_id(trans_id)
        if to_file_id is not None:
            to_trans_ids[to_file_id] = trans_id
    return (from_trans_ids, to_trans_ids)