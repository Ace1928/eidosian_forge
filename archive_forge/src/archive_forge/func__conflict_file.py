import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
def _conflict_file(self, name, parent_id, path, tree, suffix, lines=None, filter_tree_path=None):
    """Emit a single conflict file."""
    name = name + '.' + suffix
    trans_id = self.tt.create_path(name, parent_id)
    transform.create_from_tree(self.tt, trans_id, tree, path, chunks=lines, filter_tree_path=filter_tree_path)
    return trans_id