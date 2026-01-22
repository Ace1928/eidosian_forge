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
def _unversioned_parents(self, by_parent):
    """If parent directories are versioned, children must be versioned."""
    for parent_id, children in by_parent.items():
        if parent_id == ROOT_PARENT:
            continue
        if self.final_is_versioned(parent_id):
            continue
        for child_id in children:
            if self.final_is_versioned(child_id):
                yield ('unversioned parent', parent_id)
                break