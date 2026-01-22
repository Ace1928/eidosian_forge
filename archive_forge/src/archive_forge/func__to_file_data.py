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
def _to_file_data(self, to_trans_id, from_trans_id, from_executable):
    """Get data about a file in the to (target) state

        Return a (name, parent, kind, executable) tuple
        """
    to_name = self.final_name(to_trans_id)
    to_kind = self.final_kind(to_trans_id)
    to_parent = self.final_file_id(self.final_parent(to_trans_id))
    if to_trans_id in self._new_executability:
        to_executable = self._new_executability[to_trans_id]
    elif to_trans_id == from_trans_id:
        to_executable = from_executable
    else:
        to_executable = False
    return (to_name, to_parent, to_kind, to_executable)