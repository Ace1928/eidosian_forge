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
def _dump_conflicts(self, name, paths, parent_id, lines=None, no_base=False):
    """Emit conflict files.
        If this_lines, base_lines, or other_lines are omitted, they will be
        determined automatically.  If set_version is true, the .OTHER, .THIS
        or .BASE (in that order) will be created as versioned files.
        """
    base_path, other_path, this_path = paths
    if lines:
        base_lines, other_lines, this_lines = lines
    else:
        base_lines = other_lines = this_lines = None
    data = [('OTHER', self.other_tree, other_path, other_lines), ('THIS', self.this_tree, this_path, this_lines)]
    if not no_base:
        data.append(('BASE', self.base_tree, base_path, base_lines))
    if self.this_tree.supports_content_filtering():
        filter_tree_path = this_path
    else:
        filter_tree_path = None
    file_group = []
    for suffix, tree, path, lines in data:
        if path is not None:
            trans_id = self._conflict_file(name, parent_id, path, tree, suffix, lines, filter_tree_path)
            file_group.append(trans_id)
    return file_group