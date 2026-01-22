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
def _content_match(tree, entry, tree_path, kind, target_path):
    if entry.kind != kind:
        return False
    if entry.kind == 'directory':
        return True
    if entry.kind == 'file':
        with open(target_path, 'rb') as f1, tree.get_file(tree_path) as f2:
            if osutils.compare_files(f1, f2):
                return True
    elif entry.kind == 'symlink':
        if tree.get_symlink_target(tree_path) == os.readlink(target_path):
            return True
    return False